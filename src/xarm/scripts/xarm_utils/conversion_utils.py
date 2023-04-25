import numpy as np
import torch
import torchvision.transforms as T
import numpy as np
import cv2

import tf.transformations as ttf

def to_torch(array, device="cpu"):
    if isinstance(array, torch.Tensor):
        return array.to(device)
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device)
    else:
        return torch.tensor(array).to(device)

def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array

def preproc_obs(rgb, depth, camera_poses, K_matrices, state):
    H, W = rgb.shape[-2:]
    sq_size = min(H, W)

    rgb = rgb[..., :sq_size, :sq_size]
    rgb = rgb.transpose([1, 2, 0])
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.transpose([2, 0, 1])

    depth = depth[..., :sq_size, :sq_size]
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_LINEAR)
    depth = depth.reshape([1, 224, 224])

    # for depth, a value of 1.0 is 1 meter away. let's crop it to 2 meters, and handle nans and infs.
    depth[np.isnan(depth)] = 0.0
    depth[np.isinf(depth)] = 0.0
    depth[depth > 2.0] = 2.0

    # resize according to left crop / scale
    K_matrices[:2, :] *= 224.0 / sq_size

    # convert the state to a Pose object and back,
    # which will ensure that the quaternion is normalized and positive scalar.
    state = Pose(*state).to_numpy()

    obs = {}
    obs['rgb'] = rgb
    obs['depth'] = depth
    obs['camera_poses'] = camera_poses
    obs['K_matrices'] = K_matrices
    obs['state'] = state

    return obs

class Pose(object):
    def __init__(self, x, y, z, qw, qx, qy, qz):
        self.p = np.array([x, y, z])

        # we internally use tf.transformations, which uses [x, y, z, w] for quaternions.
        self.q = np.array([qx, qy, qz, qw])

        # make sure that the quaternion has positive scalar part
        if self.q[3] < 0:
            self.q *= -1

        self.q = self.q / np.linalg.norm(self.q)

    def __mul__(self, other):
        assert isinstance(other, Pose)
        p = self.p + ttf.quaternion_matrix(self.q)[:3, :3].dot(other.p)
        q = ttf.quaternion_multiply(self.q, other.q)
        return Pose(p[0], p[1], p[2], q[3], q[0], q[1], q[2])

    def __rmul__(self, other):
        assert isinstance(other, Pose)
        return other * self

    def __str__(self):
        return "p: {}, q: {}".format(self.p, self.q)

    def inv(self):
        R = ttf.quaternion_matrix(self.q)[:3, :3]
        p = -R.T.dot(self.p)
        q = ttf.quaternion_inverse(self.q)
        return Pose(p[0], p[1], p[2], q[3], q[0], q[1], q[2])

    def to_numpy(self):
        """
        this satisfies Pose(*to_numpy(p)) == p
        """
        q_reverted = np.array([self.q[3], self.q[0], self.q[1], self.q[2]])
        return np.concatenate([self.p, q_reverted])

    def to_axis_angle(self):
        """
        returns the axis-angle representation of the rotation.
        """
        angle = 2 * np.arccos(self.q[3])
        if angle > np.pi:
            angle -= 2 * np.pi

        axis = self.q[:3] / np.linalg.norm(self.q[:3])
        p = self.p

        return np.concatenate([p, axis, [angle]])

    def to_44_matrix(self):
        out = np.eye(4)
        out[:3, :3] = ttf.quaternion_matrix(self.q)[:3, :3]
        out[:3, 3] = self.p
        return out

    @staticmethod
    def from_axis_angle(x, y, z, ax, ay, az, phi):
        """
        returns a Pose object from the axis-angle representation of the rotation.
        """

        p = np.array([x, y, z])
        qw = np.cos(phi / 2.0)
        qx = ax * np.sin(phi / 2.0)
        qy = ay * np.sin(phi / 2.0)
        qz = az * np.sin(phi / 2.0)

        return Pose(p[0], p[1], p[2], qw, qx, qy, qz)

def compute_inverse_action(p, p_new, ee_control=False, scale_factor=None):
    assert isinstance(p, Pose) and isinstance(p_new, Pose)
    if ee_control:
        dpose = p.inv() * p_new
    else:
        dpose = p_new * p.inv()

    if scale_factor is not None:
        dpose.p = dpose.p / scale_factor

    return dpose

def compute_forward_action(p, dpose, ee_control=False, scale_factor=None):
    assert isinstance(p, Pose) and isinstance(dpose, Pose)
    dpose = Pose(*dpose.to_numpy())
    if scale_factor is not None:
        dpose.p = dpose.p * scale_factor

    if ee_control:
        p_new = p * dpose
    else:
        p_new = dpose * p

    return p_new
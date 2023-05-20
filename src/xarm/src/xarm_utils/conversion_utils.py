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

def preproc_obs(rgb, depth, camera_poses, K_matrices, state, rotation_mode='quat', rgb_base=None, depth_base=None):
    H, W = rgb.shape[-2:]
    sq_size = min(H, W)

    # crop the center square
    if H > W:
        rgb = rgb[..., (-sq_size/2):(sq_size/2), :sq_size]
        depth = depth[..., (-sq_size/2):(sq_size/2), :sq_size]
        K_matrices[:2, :] *= 224.0 / sq_size # TODO: K-matrix has now changed because of a center crop. Figure out the change.
    elif W < H:
        rgb = rgb[..., :sq_size, (-sq_size/2):(sq_size/2)]
        depth = depth[..., :sq_size, (-sq_size/2):(sq_size/2)]
        K_matrices[:2, :] *= 224.0 / sq_size # TODO: K-matrix has now changed because of a center crop. Figure out the change.

    rgb = rgb.transpose([1, 2, 0])
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.transpose([2, 0, 1])

    # save as a uint8 for memory efficiency. it is currently a float64 in 0-255.
    rgb = rgb.astype(np.uint8)

    if rgb_base is not None:
        rgb_base = rgb_base.transpose([1, 2, 0])
        rgb_base = cv2.resize(rgb_base, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb_base = rgb_base.transpose([2, 0, 1])

        rgb = np.stack([rgb, rgb_base], axis=0)

    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_LINEAR)
    depth = depth.reshape([1, 224, 224])

    # for depth, a value of 1.0 is 1 meter away. let's crop it to 2 meters, and handle nans and infs.
    depth[np.isnan(depth)] = 0.0
    depth[np.isinf(depth)] = 0.0
    depth[depth > 2.0] = 2.0

    if depth_base is not None:
        depth_base = cv2.resize(depth_base, (224, 224), interpolation=cv2.INTER_LINEAR)
        depth_base = depth_base.reshape([1, 224, 224])
        depth_base[np.isnan(depth_base)] = 0.0
        depth_base[np.isinf(depth_base)] = 0.0
        depth_base[depth_base > 2.0] = 2.0

        depth = np.stack([depth, depth_base], axis=0)

    # convert the state to a Pose object and back,
    # which will ensure that the quaternion is normalized and positive scalar.
    if rotation_mode == "quat":
        state = Pose(*state).to_quaternion()
    elif rotation_mode == "aa":
        state = Pose(*state).to_axis_angle()
    elif rotation_mode == "euler":
        state = Pose(*state).to_euler()
    else:
        raise ValueError("invalid rotation mode")
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

    def to_quaternion(self):
        """
        this satisfies Pose(*to_quaternion(p)) == p
        """
        q_reverted = np.array([self.q[3], self.q[0], self.q[1], self.q[2]])
        return np.concatenate([self.p, q_reverted])

    def to_axis_angle(self):
        """
        returns the axis-angle representation of the rotation.
        """
        angle = 2 * np.arccos(self.q[3])
        angle = angle / np.pi
        if angle > 1:
            angle -= 2

        axis = self.q[:3] / np.linalg.norm(self.q[:3])

        # keep the axes positive
        if axis[0] < 0:
            axis *= -1
            angle *= -1

        return np.concatenate([self.p, axis, [angle]])

    def to_euler(self):
        q = np.array(ttf.euler_from_quaternion(self.q))
        if q[0] > np.pi:
            q[0] -= 2 * np.pi
        if q[1] > np.pi:
            q[1] -= 2 * np.pi
        if q[2] > np.pi:
            q[2] -= 2 * np.pi

        q = q / np.pi

        return np.concatenate([self.p, q, [0.0]])

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

        phi = phi * np.pi
        p = np.array([x, y, z])
        qw = np.cos(phi / 2.0)
        qx = ax * np.sin(phi / 2.0)
        qy = ay * np.sin(phi / 2.0)
        qz = az * np.sin(phi / 2.0)

        return Pose(p[0], p[1], p[2], qw, qx, qy, qz)

    @staticmethod
    def from_euler(x, y, z, roll, pitch, yaw, _):
        """
        returns a Pose object from the euler representation of the rotation.
        """
        p = np.array([x, y, z])
        roll, pitch, yaw = roll * np.pi, pitch * np.pi, yaw * np.pi
        q = ttf.quaternion_from_euler(roll, pitch, yaw)
        return Pose(p[0], p[1], p[2], q[3], q[0], q[1], q[2])

    @staticmethod
    def from_quaternion(x, y, z, qw, qx, qy, qz):
        """
        returns a Pose object from the quaternion representation of the rotation.
        """
        p = np.array([x, y, z])
        return Pose(p[0], p[1], p[2], qw, qx, qy, qz)

def compute_inverse_action(p, p_new, ee_control=False):
    assert isinstance(p, Pose) and isinstance(p_new, Pose)

    if ee_control:
        dpose = p.inv() * p_new
    else:
        dpose = p_new * p.inv()

    return dpose

def compute_forward_action(p, dpose, ee_control=False):
    assert isinstance(p, Pose) and isinstance(dpose, Pose)
    dpose = Pose.from_quaternion(*dpose.to_quaternion())

    if ee_control:
        p_new = p * dpose
    else:
        p_new = dpose * p

    return p_new
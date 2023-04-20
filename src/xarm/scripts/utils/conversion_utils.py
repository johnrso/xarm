import numpy as np
import torch
import torchvision.transforms as T
import numpy as np

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
    image_transform = T.Compose([
        T.Resize((224, 224), T.InterpolationMode.BILINEAR, antialias=None),
    ])

    H, W = rgb.shape[-2:]
    sq_size = min(H, W)

    rgb_t = to_torch(rgb, device='cuda')
    rgb_t = rgb_t[..., :sq_size, :sq_size] # TODO: left crop?
    rgb_t = image_transform(rgb_t)
    rgb = to_numpy(rgb_t)

    depth_t = to_torch(depth, device='cuda')
    depth_t = depth_t.unsqueeze(-3)
    depth_t = depth_t[..., :sq_size, :sq_size] # TODO: left crop?
    depth_t = image_transform(depth_t)
    depth = to_numpy(depth_t)

    depth[np.isnan(depth)] = 0.0

    # resize according to left crop / scale
    K_matrices[:2, :] *= 224.0 / sq_size

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
        self.q = np.array([qx, qy, qz, qw])

        # make sure that the quaternion has positive scalar part
        if self.q[3] < 0:
            self.q *= -1

        dpose.q = dpose.q / np.linalg.norm(dpose.q)

    def __mul__(self, other):
        assert isinstance(other, Pose)
        p = self.p + ttf.quaternion_matrix(self.q)[:3, :3].dot(other.p)
        q = ttf.quaternion_multiply(self.q, other.q)
        return Pose(p[0], p[1], p[2], q[3], q[0], q[1], q[2])

    def __rmul__(self, other):
        assert isinstance(other, Pose)
        return other * self

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

    def to_44_matrix(self):
        out = np.eye(4)
        out[:3, :3] = ttf.quaternion_matrix(self.q)[:3, :3]
        out[:3, 3] = self.p
        return out

    def __str__(self):
        return "p: {}, q: {}".format(self.p, self.q)


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
    if scaled_factor is not None:
        dpose.p = dpose.p * scale_factor


    if ee_control:
        p_new = p * dpose
    else:
        p_new = dpose * p

    return p_new
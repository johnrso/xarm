from gdict.data import GDict
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

    _, H, W = rgb.shape
    sq_size = min(H, W)

    rgb_t = to_torch(rgb, device='cuda')
    rgb_t = rgb_t[:, :sq_size, :sq_size] # TODO: left crop?
    rgb_t = image_transform(rgb_t)
    rgb = to_numpy(rgb_t)

    depth_t = to_torch(depth, device='cuda')
    depth_t = depth_t.view(1, *depth_t.shape)
    depth_t = depth_t[:, :sq_size, :sq_size] # TODO: left crop?
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
        return np.concatenate([self.p, self.q])
    
def compute_inverse_action(p, p_new, ee_control=False, scaled_actions=False):
    assert isinstance(p, Pose) and isinstance(p_new, Pose)
    if ee_control:
        dpose = p.inv() * p_new
    else:
        dpose = p_new * p.inv()

    if scaled_actions:
        dpose.p = dpose.p / 0.1

    return dpose

def compute_forward_action(p, dpose, ee_control=False, scaled_actions=False):
    assert isinstance(p, Pose) and isinstance(dpose, Pose)
    if scaled_actions:
        dpose.p = dpose.p * 0.1

    if ee_control:
        p_new = p * dpose
    else:
        p_new = dpose * p

    return p_new
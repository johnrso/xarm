from gdict.data import GDict
import torch
import torch.transforms as T
import numpy as np

from tf.transforms import Quaternion

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
    obs = GDict()
    image_transform = T.Compose([
        T.CenterCrop(224),
        T.Resize((224, 224), T.InterpolationMode.BILINEAR)
    ])

    _, H, W = rgb.shape
    sq_size = min(H, W)

    rgb_t = to_torch(rgb, device='cuda')
    rgb_t = rgb_t[:, :sq_size, :sq_size] # TODO: left crop?
    rgb_t = image_transform(rgb_t)
    rgb = to_numpy(rgb_t)

    depth_t = to_torch(depth, device='cuda')
    depth_t = depth_t.view(1, **depth_t.shape)
    depth_t = depth_t[:, :sq_size, :sq_size] # TODO: left crop?
    depth_t = image_transform(depth_t)
    depth = to_numpy(depth_t)

    depth[np.is_nan(depth)] = 0.0

    # resize according to left crop / scale
    K_matrices[:2, :] *= 224.0 / sq_size

    obs['rgb'] = rgb
    obs['depth'] = depth
    obs['camera_poses'] = camera_poses
    obs['K_matrices'] = K_matrices
    obs['state'] = state
    return obs

class Pose(object):
    def __init__(self, p, q):
        self.p = p
        self.q = q

        assert isinstance(self.q, Quaternion)

    def __mul__(self, other):
        return Pose(self.p + self.q.rotate(other.p), self.q * other.q)

    def __rmul__(self, other):
        return Pose(other.q.rotate(self.p) + other.p, other.q * self.q)

    def __truediv__(self, other):
        return Pose(self.p - self.q.rotate(other.p), self.q / other.q)

    def __rtruediv__(self, other):
        return Pose(other.q.rotate(self.p) - other.p, other.q / self.q)

    def __str__(self):
        return "p: {}, q: {}".format(self.p, self.q)

    def inv(self):
        return Pose(-self.q.inverse.rotate(self.p), self.q.inverse())

def compute_inverse_action(p, p_new, ee_frame_control=False, scaled_actions=False):
    assert isinstance(p, Pose) and isinstance(p_new, Pose)
    if ee_frame_control:
        dpose = p.inv() * p_new
    else:
        dpose = p_new * p.inv()

    if scaled_actions:
        dpose = dpose / 0.1

    return dpose

def compute_forward_action(p, dpose, ee_frame_control=False, scaled_actions=False):
    assert isinstance(p, Pose) and isinstance(dpose, Pose)
    if scaled_actions:
        dpose = dpose * 0.1

    if ee_frame_control:
        p_new = p * dpose
    else:
        p_new = dpose * p

    return p_new
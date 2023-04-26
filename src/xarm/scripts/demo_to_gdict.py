# observations needed:
# 'obs':
    # CHW 'rgb', CHW 'depth', 4x4 'camera_poses', 'state': stack of EE poses, 'K_matrices'': stack of K matrices.
# 'actions':
    # Record delta pose of EE.
# 'dones':
    # Just spam a bunch of falses, then a true.
# 'episode_dones':
    # Just spam a bunch of falses, then a true.


# Pickle to GDict mappings:
# 'rgb': 'rgb'
# 'depth': 'depth'
# 'camera_poses': 'T_camera_in_link0'
# 'state': 'p_ee_in_link0'
# 'K_matrices': 'K'
# 'actions': just take state and do subtractions??
# 'dones' and 'episode_dones' are arbitrary

import glob
import os
import pickle
import shutil
from tqdm import tqdm

from natsort import natsorted
import numpy as np
import hydra

from omegaconf import DictConfig, OmegaConf

from gdict.data import GDict, DictArray
from xarm_utils.conversion_utils import preproc_obs, Pose, compute_inverse_action, compute_forward_action

import mediapy as mp
import matplotlib.pyplot as plt

from simple_bc.utils.visualization_utils import make_grid_video_from_numpy

def get_act_bounds(source_dir, i, ee_control=False, rotation_mode='quat'):
    pkls = natsorted(glob.glob(os.path.join(source_dir, '**/*.pkl'), recursive=True), reverse=True)
    demo_stack = []

    # go through the demo in reverse order.
    requested_control = None # this stores the requested pose of time t+1.
    scale_factor = None

    if len(pkls) <= 30:
        print(f"Skipping {source_dir} because it has less than 30 frames.")
        return np.zeros(3)

    for pkl in pkls:
        curr_ts = {}
        with open(pkl, 'rb') as f:
            demo = pickle.load(f)

        view = "wrist"
        rgb = demo.pop(f'rgb_{view}').transpose([2, 0, 1]) * 1.0
        depth = demo.pop(f'depth_{view}')
        K = demo.pop(f'K_{view}')
        T_camera_in_link0 = demo.pop('T_camera_in_link0')
        p_ee_in_link0 = demo.pop('p_ee_in_link0')

        obs = preproc_obs(rgb=rgb,
                          depth=depth,
                          camera_poses=T_camera_in_link0,
                          K_matrices=K,
                          state=p_ee_in_link0,
                          rotation_mode=rotation_mode)

        curr_ts['obs'] = obs

        if requested_control is not None:
            curr_pose = obs['state']

            # Compute transform from previous state to current state.
            if rotation_mode == "quat":
                cpose = Pose.from_quaternion(*curr_pose)
                rpose = Pose.from_quaternion(*requested_control)
            elif rotation_mode == "aa":
                cpose = Pose.from_axis_angle(*curr_pose)
                rpose = Pose.from_quaternion(*requested_control)
            elif rotation_mode == "euler":
                cpose = Pose.from_euler(*curr_pose)
                rpose = Pose.from_quaternion(*requested_control)
            action = compute_inverse_action(cpose, rpose, ee_control=ee_control)
            if scale_factor is None:
                scale_factor = action.p
            else:
                scale_factor = np.maximum(scale_factor, action.p)

        requested_control = demo.pop('control')

    assert scale_factor is not None

    return scale_factor

def convert_single_demo(source_dir,
                        i,
                        traj_output_dir,
                        rgb_output_dir,
                        depth_output_dir,
                        state_output_dir,
                        action_output_dir,
                        ee_control=False,
                        scale_factor=-1,
                        cleanup_gripper=False,
                        rotation_mode="quat"):
    """
    1. converts the demo into a gdict
    2. visualizes the RGB of the demo
    3. visualizes the state + action space of the demo
    4. returns these to be collated by the caller.
    """

    pkls = natsorted(glob.glob(os.path.join(source_dir, '**/*.pkl'), recursive=True), reverse=True)
    demo_stack = []

    view = "wrist" # TODO: support more views eventually
    if len(pkls) <= 30:
        return 0

    # go through the demo in reverse order.
    requested_control = None # this stores the requested pose of time t+1.
    # remove the first few frames because they are not useful.
    pkls = pkls[:-5]

    for pkl in pkls:
        curr_ts = {}

        with open(pkl, 'rb') as f:
            demo = pickle.load(f)

        rgb = demo.pop(f'rgb_{view}').transpose([2, 0, 1]) * 1.0
        depth = demo.pop(f'depth_{view}')
        K = demo.pop(f'K_{view}')
        T_camera_in_link0 = demo.pop('T_camera_in_link0')
        p_ee_in_link0 = demo.pop('p_ee_in_link0')

        obs = preproc_obs(rgb=rgb,
                          depth=depth,
                          camera_poses=T_camera_in_link0,
                          K_matrices=K,
                          state=p_ee_in_link0,
                          rotation_mode=rotation_mode)

        curr_ts['obs'] = obs

        if requested_control is not None:
            curr_pose = obs['state']
            gripper_state = demo.pop('gripper_state')

            rpose = Pose.from_quaternion(*requested_control)
            if rotation_mode == "quat":
                cpose = Pose.from_quaternion(*curr_pose)
                action = compute_inverse_action(cpose, rpose, ee_control=ee_control, scale_factor=scale_factor)
                act = action.to_quaternion()
            elif rotation_mode == "aa":
                cpose = Pose.from_axis_angle(*curr_pose)
                action = compute_inverse_action(cpose, rpose, ee_control=ee_control, scale_factor=scale_factor)
                act = action.to_axis_angle()
            elif rotation_mode == "euler":
                cpose = Pose.from_euler(*curr_pose)
                action = compute_inverse_action(cpose, rpose, ee_control=ee_control, scale_factor=scale_factor)
                act = action.to_euler()
            else:
                raise NotImplementedError
            curr_ts['actions'] = np.concatenate([act, [gripper_state]])
        else:
            # if this is the last frame, then we don't have a requested control. Just set it to the current pose.
            action = np.zeros(8)
            # action[3] = 1.0
            action[7] = demo.pop('gripper_state')

            curr_ts['actions'] = action

        requested_control = demo.pop('control')
        curr_ts['dones'] = np.zeros(1) # random fill
        curr_ts['episode_dones'] = np.zeros(1) # random fill

        curr_ts_wrapped = dict()
        curr_ts_wrapped[f'traj_{i}'] = curr_ts

        demo_stack = [curr_ts_wrapped] + demo_stack

    if cleanup_gripper: # convert first repeated instance of 'closed', 'closed', ... to 'open', 'open', ... (i.e. start the demo as open)
        for curr_ts in demo_stack:
            if curr_ts[f'traj_{i}']['actions'][7] == 0.0:
                break
            else:
                curr_ts[f'traj_{i}']['actions'][7] = 0.0


    demo_dict = DictArray.stack(demo_stack)
    GDict.to_hdf5(demo_dict, os.path.join(traj_output_dir + "", f'traj_{i}.h5'))

    # save the rgb and depth videos
    all_rgbs = demo_dict[f'traj_{i}']['obs']['rgb'].transpose([0, 2, 3, 1])
    all_rgbs = all_rgbs.astype(np.uint8)
    _, H, W, _ = all_rgbs.shape
    all_depths = demo_dict[f'traj_{i}']['obs']['depth'].reshape([-1, H, W])
    all_depths = all_depths / 2.0 # scale to 0-1

    mp.write_video(os.path.join(rgb_output_dir + "", f'traj_{i}_rgb.mp4'), all_rgbs, fps=30)
    mp.write_video(os.path.join(depth_output_dir + "", f'traj_{i}_depth.mp4'), all_depths, fps=30)

    all_depths = np.tile(all_depths[..., None], [1, 1, 1, 3])

    # save the state and action plots
    all_actions = demo_dict[f'traj_{i}']['actions']
    all_states = demo_dict[f'traj_{i}']['obs']['state']

    curr_actions = all_actions.reshape([1, *all_actions.shape])
    curr_states = all_states.reshape([-1, *all_states.shape])

    plot_in_grid(curr_actions, os.path.join(action_output_dir + "", f'traj_{i}_actions.png'))
    plot_in_grid(curr_states, os.path.join(state_output_dir + "", f'traj_{i}_states.png'))

    return all_rgbs, all_depths, all_actions, all_states

def plot_in_grid(vals, save_path):
    """
    vals: B x T x N, where
    B is the number of trajectories,
    T is the number of timesteps,
    N is the dimensionality of the values.
    """
    B = len(vals)
    N = vals[0].shape[-1]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for b in range(B):
        curr = vals[b]
        for i in range(N):
            T = curr.shape[0]
            # give them transparency
            axes[i // 4, i % 4].plot(np.arange(T), curr[:, i], alpha=0.25)

    for i in range(N):
        axes[i // 4, i % 4].set_title(f"Dim {i}")

    plt.savefig(save_path)
    plt.close()

@hydra.main(config_path='../conf/data', config_name='no_ee')
def main(cfg):
    subdirs = natsorted(glob.glob(os.path.join(cfg.source_dir, '*/'), recursive=True))

    output_dir = cfg.source_dir
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    output_dir += "_conv"
    output_dir += ("_ee" if cfg.ee_control else "")
    output_dir += ("_cleangrip" if cfg.cleanup_gripper else "")
    output_dir += f"_{cfg.rotation_mode}"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        input(f"Output directory {output_dir} already exists, and will be deleted. Press enter to continue...")
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)


    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)

    val_size = int(min(0.1 * len(subdirs), 5))
    val_indices = np.random.choice(len(subdirs), size=val_size, replace=False)
    val_indices = set(val_indices)

    if cfg.scale_factor != "none":
        scale_factor = np.array(cfg.scale_factor)
    else:
        pbar = tqdm(range(len(subdirs)))
        scale_factor = None
        for i in pbar:
            # if i in val_indices:
            #     continue
            curr_scale_factor = get_act_bounds(subdirs[i], i, ee_control=cfg.ee_control, rotation_mode=cfg.rotation_mode)
            if scale_factor is None:
                scale_factor = curr_scale_factor
            else:
                scale_factor = np.maximum(scale_factor, curr_scale_factor)
            pbar.set_description(f"t: {i}")

    print(f"Scale factor is {scale_factor.tolist()}. Outputting to {output_dir}. (val indices: {val_indices}))")
    pbar = tqdm(range(len(subdirs)))
    tot = 0

    all_rgbs = []
    all_depths = []
    all_actions = []
    all_states = []

    vis_dir = os.path.join(output_dir, 'vis')
    state_output_dir = os.path.join(vis_dir, 'state')
    action_output_dir = os.path.join(vis_dir, 'action')
    rgb_output_dir = os.path.join(vis_dir, 'rgb')
    depth_output_dir = os.path.join(vis_dir, 'depth')

    if not os.path.isdir(vis_dir):
        os.mkdir(vis_dir)
    if not os.path.isdir(state_output_dir):
        os.mkdir(state_output_dir)
    if not os.path.isdir(action_output_dir):
        os.mkdir(action_output_dir)
    if not os.path.isdir(rgb_output_dir):
        os.mkdir(rgb_output_dir)
    if not os.path.isdir(depth_output_dir):
        os.mkdir(depth_output_dir)

    for i in pbar:
        out_dir = val_dir if i in val_indices else train_dir
        out_dir = os.path.join(out_dir, "none")

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        ret = convert_single_demo(subdirs[i],
                                  i,
                                  traj_output_dir=out_dir,
                                  rgb_output_dir=rgb_output_dir,
                                  depth_output_dir=depth_output_dir,
                                  state_output_dir=state_output_dir,
                                  action_output_dir=action_output_dir,
                                  ee_control=cfg.ee_control,
                                  rotation_mode=cfg.rotation_mode,
                                  scale_factor=scale_factor,
                                  cleanup_gripper=cfg.cleanup_gripper)
        if ret != 0:
            all_rgbs.append(ret[0])
            all_depths.append(ret[1])
            all_actions.append(ret[2])
            all_states.append(ret[3])
            tot += 1

        pbar.set_description(f"t: {i}")
    print(f"Finished converting all demos to {output_dir}! (num demos: {tot} / {len(subdirs)})")

    cfg.scale_factor = scale_factor.tolist()

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)

    if len(all_rgbs) > 0:
        print(f"Visualizing all demos...")

        make_grid_video_from_numpy(all_rgbs, 10, os.path.join(rgb_output_dir, "_all_rgb.mp4"), fps=30)
        make_grid_video_from_numpy(all_depths, 10, os.path.join(depth_output_dir, "_all_depth.mp4"), fps=30)
        plot_in_grid(all_actions, os.path.join(action_output_dir, "_all_actions.png"))
        plot_in_grid(all_states, os.path.join(state_output_dir, "_all_states.png"))

if __name__ == '__main__':
    main()

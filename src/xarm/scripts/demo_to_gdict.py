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
from tqdm import tqdm

from natsort import natsorted
import numpy as np
import hydra

from gdict.data import GDict, DictArray
from utils.conversion_utils import preproc_obs, Pose, compute_inverse_action

def convert_single_demo(source_dir, i, output_dir, ee_control=False, scaled_actions=False):
    pkls = natsorted(glob.glob(os.path.join(source_dir, '**/*.pkl'), recursive=True), reverse=True)
    demo_stack = []

    view = "wrist" # TODO: support more views eventually
    if len(pkls) <= 60:
        print(f"Skipping {source_dir} because it has less than 30 frames.")
        return 0

    # go through the demo in reverse order.
    requested_control = None # this stores the requested pose of time t+1.
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
                          state=p_ee_in_link0)
        
        curr_ts['obs'] = obs

        if requested_control is not None:
            curr_pose = obs['state']
            gripper_state = demo.pop('gripper_state')

            # Compute transform from previous state to current state.
            cpose = Pose(*curr_pose)
            rpose = Pose(*requested_control)

            action = compute_inverse_action(cpose, rpose, ee_control=ee_control, scaled_actions=scaled_actions) 
            curr_ts['actions'] = np.concatenate([action.to_numpy(), [gripper_state]])
        else:
            # if this is the last frame, then we don't have a requested control. Just set it to the current pose.
            action = np.zeros(8)
            action[3] = 1.0
            action[7] = demo.pop('gripper_state')

            curr_ts['actions'] = action

        requested_control = demo.pop('control')
        curr_ts['dones'] = np.zeros(1) # random fill
        curr_ts['episode_dones'] = np.zeros(1) # random fill

        curr_ts_wrapped = dict()
        curr_ts_wrapped[f'traj_{i}'] = curr_ts
        demo_stack.append(curr_ts_wrapped)

    # Save the demo.
    demo_dict = DictArray.stack(demo_stack)
    GDict.to_hdf5(demo_dict, os.path.join(output_dir + "", f'traj_{i}.h5'))

    return 1

@hydra.main(config_path='../conf/data', config_name='scaled_no_ee')
def main(cfg):
    subdirs = natsorted(glob.glob(os.path.join(cfg.source_dir, '*/'), recursive=True))

    output_dir = cfg.source_dir
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    output_dir += ("_e" if cfg.ee_control else "")
    output_dir += ("_s" if cfg.scaled_actions else "")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_dir = os.path.join(output_dir, 't')
    val_dir = os.path.join(output_dir, 'v')

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)


    val_indices = np.random.choice(len(subdirs), size=5, replace=False)
    val_indices = set(val_indices)

    print(f"Outputting to {output_dir}. (val indices: {val_indices}))")

    pbar = tqdm(range(len(subdirs)))
    tot = 0
    for i in pbar:
        out_dir = val_dir if i in val_indices else train_dir
        tot += convert_single_demo(subdirs[i], i, output_dir=out_dir, ee_control=cfg.ee_control, scaled_actions=cfg.scaled_actions)
        pbar.set_description(f"t: {i}")
    
    print(f"Finished converting all demos to {output_dir}! (num demos: {tot} / {len(subdirs)})")

if __name__ == '__main__':
    main()
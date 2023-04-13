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

import yaml

import torch
from torchvision import transforms
from natsort import natsorted
import numpy as np
import hydra

from gdict.data import GDict, DictArray
from utils.conversion_utils import preproc_obs

def convert_single_demo(source_dir, i, output_dir, use_mouse=False, view="wrist", ee_frame_control=False, scaled_actions=False):
    print(source_dir)
    pkls = natsorted(glob.glob(os.path.join(source_dir, '**/*.pkl'), recursive=True))
    demo_stack = []

    if len(pkls) <= 30:
        print(f"Skipping {source_dir} because it has less than 30 frames.")

    for pkl in pkls:
        # Load the demo.
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

        if len(demo_stack) > 0:
            prev_pose = demo_stack[-1][f'traj_{i}']['obs']['state']
            curr_pose = demo_dict['obs']['state']
            gripper_state = demo.pop('gripper_state')
            mouse_control = demo.pop('control')

            # Compute transform from previous state to current state.
            pose = Pose(prev_pose[:3], prev_pose[3:])

            if use_mouse:
                pose_new = Pose(mouse_control[:3], mouse_control[3:])
            else:
                pose_new = Pose(curr_pose[:3], curr_pose[3:])

            if ee_frame_control:
                delta_pose = pose.inv() * pose_new
            else:
                delta_pose = pose_new * pose.inv()

            if scaled_actions:

                p = delta_pose.p / 0.1
                q = delta_pose.q
            else:
                p = delta_pose.p
                q = delta_pose.q
            demo_stack[-1][f'traj_{i}']['actions'] = np.concatenate([p, q, [gripper_state]]) # store EE delta pose
        demo_dict['dones'] = np.zeros(1) # random fill
        demo_dict['episode_dones'] = np.zeros(1) # random fill

        demo_dict_final = dict()
        demo_dict_final[f'traj_{i}'] = demo_dict
        demo_stack.append(demo_dict_final)

    # Final action delta is identity; no change in EE pose. Gripper is same position as previous action.
    final_action = np.zeros(8)
    final_action[3] = 1.0
    final_action[7] = demo_stack[-2][f'traj_{i}']['actions'][7]
    demo_stack[-1][f'traj_{i}']['actions'] = final_action

    # Save the demo.
    demo_dict = DictArray.stack(demo_stack)
    print("Demo dict shape for verification:", demo_dict.shape)
    GDict.to_hdf5(demo_dict, os.path.join(output_dir + "", f'traj_{i}.h5'))

    print(f"Finished converting demo trajectory {i}")

@hydra.main(config_path='../config', config_name='data')
def main(cfg):
    subdirs = natsorted(glob.glob(os.path.join(args.source_dir, '*/'), recursive=True))

    if args.output_dir is None:
        output_dir = args.source_dir
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]

        output_dir += "_" if not args.use_mouse else "_m"
        output_dir += "_" + args.view[0]
        output_dir += ("_e" if args.ee_frame_control else "")
        output_dir += ("_s" if args.scaled_actions else "")
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i in range(len(subdirs)):
        convert_single_demo(subdirs[i], i, output_dir=output_dir, use_mouse=args.use_mouse, view=args.view, ee_frame_control=args.ee_frame_control, scaled_actions=args.scaled_actions)
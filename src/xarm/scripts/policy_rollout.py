#!/usr/bin/env python
import os
import sys
import time

import numpy as np
from gdict.data import GDict

import skrobot
import torch
import torchvision.transforms as transforms
from einops import rearrange

import cv_bridge
import message_filters
import rospy
import tf
import tf.transformations as ttf

from sensor_msgs.msg import CameraInfo, Image, JointState

import time

import xarm_ws.src.xarm.scripts.utils.robot_utils as robot_utils
import click
import wandb

from omegaconf import OmegaConf

class Agent:
    def __init__(self,
                 policy):

        self.vid = []
        self.all_vid = []

        self.num_iter = 0

        # policy setup
        rospy.init_node("agent", anonymous=True)
        rospy.on_shutdown(self.close)

        self._setup_robot()
        self._setup_listeners()

    def _setup_robot(self):
        self._gripper_action_prev = True  # true if closed, false if open
        self._robot = robot_utils.XArm()
        self._ri = robot_utils.XArmROSRobotInterface(self._robot, namespace="")
        self._ri.update_robot_state()
        self._robot.angle_vector(self._ri.angle_vector())
    
    def _setup_listeners(self):
        self._tf_listener = tf.listener.TransformListener(cache_time=rospy.Duration(30))

        self._tf_listener.waitForTransform(
            target_frame="link_base",
            source_frame="wrist_camera_color_optical_frame",
            time=rospy.Time(0),
            timeout=rospy.Duration(10),
        )
    
        self._sub_caminfo_wrist = message_filters.Subscriber(
            "/wrist_camera/aligned_depth_to_color/camera_info", CameraInfo
        )
        self._sub_rgb_wrist = message_filters.Subscriber(
            "/wrist_camera/color/image_rect_color", Image
        )
        self._sub_depth_wrist = message_filters.Subscriber(
            "/wrist_camera/aligned_depth_to_color/image_raw", Image
        )

        # base camera
        self._sub_caminfo_base = message_filters.Subscriber(
            "/base_camera/aligned_depth_to_color/camera_info", CameraInfo
        )
        self._sub_rgb_base = message_filters.Subscriber(
            "/base_camera/color/image_rect_color", Image
        )
        self._sub_depth_base = message_filters.Subscriber(
            "/base_camera/aligned_depth_to_color/image_raw", Image
        )

        # proprio
        self._sub_joint = message_filters.Subscriber(
            "/joint_states", JointState
        )

        sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._sub_caminfo_wrist,
                self._sub_rgb_wrist,
                self._sub_depth_wrist,
                self._sub_caminfo_base,
                self._sub_rgb_base,
                self._sub_depth_base,
                self._sub_joint
            ],
            slop=0.1,
            queue_size=50,
        )
        sync.registerCallback(self._obs_callback)

        sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._sub_rgb_wrist,
                self._sub_rgb_base,
            ],
            slop=0.1,
            queue_size=50,
        )
        sync.registerCallback(self.record_video)

    def _obs_callback(
            self,
            caminfo_msg_wrist,
            rgb_msg_wrist,
            depth_msg_wrist,
            caminfo_msg_base,
            rgb_msg_base,
            depth_msg_base,
            joint_msg
            ):

        rospy.loginfo_once("obs callback registered")

        position, quaternion = self._tf_listener.lookupTransform(
            target_frame="link_base",
            source_frame="wrist_camera_color_optical_frame",
            time=rospy.Time(0),
        )
        T_camera_in_link0 = ttf.quaternion_matrix(quaternion)
        T_camera_in_link0[:3, 3] = position

        bridge = cv_bridge.CvBridge()

        # wrist camera processing
        rgb_wrist = bridge.imgmsg_to_cv2(rgb_msg_wrist, desired_encoding="rgb8")
        depth_wrist = bridge.imgmsg_to_cv2(depth_msg_wrist)
        assert rgb_wrist.dtype == np.uint8 and rgb_wrist.ndim == 3
        assert depth_wrist.dtype == np.uint16 and depth_wrist.ndim == 2
        depth_wrist = depth_wrist.astype(np.float32) / 1000

        K_wrist = np.array(caminfo_msg_wrist.K).reshape(3, 3)

        # base camera processing
        rgb_base = bridge.imgmsg_to_cv2(rgb_msg_base, desired_encoding="rgb8")
        depth_base = bridge.imgmsg_to_cv2(depth_msg_base)
        assert rgb_base.dtype == np.uint8 and rgb_base.ndim == 3
        assert depth_base.dtype == np.uint16 and depth_base.ndim == 2
        depth_base = depth_base.astype(np.float32) / 1000

        K_base = np.array(caminfo_msg_base.K).reshape(3, 3)
        joint_positions = np.array(joint_msg.position, dtype=np.float32)
        joint_velocites = np.array(joint_msg.velocity, dtype=np.float32)

        position, quaternion = self._tf_listener.lookupTransform(
            target_frame="link_base",
            source_frame="link_tcp",
            time=rospy.Time(0),
        )

        # rotate from xyzw to wxyz
        quaternion = np.array(quaternion)[[3, 0, 1, 2]]
        p_ee_in_link0 = np.concatenate([position, quaternion]) # wxyz quaternion

        self._obs = dict(
            stamp=caminfo_msg_wrist.header.stamp,
            rgb=rgb_wrist,
            depth=depth_wrist,
            state=p_ee_in_link0,
            camera_poses=T_camera_in_link0,
            K_matrices=K_wrist,
        )

    def record_video(self, rgb_msg_wrist, rgb_msg_base):
        rospy.loginfo_once(f"recording video")
        bridge = cv_bridge.CvBridge()

        # wrist camera processing
        rgb_wrist = bridge.imgmsg_to_cv2(rgb_msg_wrist, desired_encoding="rgb8")
        rgb_base = bridge.imgmsg_to_cv2(rgb_msg_base, desired_encoding="rgb8")

        H, W = rgb_wrist.shape[:2]
        sq = min(H, W)
        rgb_wrist = rgb_wrist[:sq, :sq, :]
        rgb_base = rgb_base[:sq, :sq, :]

        vid_frame = np.concatenate([rgb_wrist, rgb_base], axis=1)

        self.vid.append(vid_frame)
        self.all_vid.append(vid_frame)

    def get_obs(
        self,
        stamp: rospy.Time,
        is_first: bool = False,
    ):
        while not self._obs:
            rospy.sleep(0.001)
        assert self._obs is not None
        obs_copy = self._obs.copy()

        # crop to square
        obs_copy["rgb"] = torch.as_tensor(obs_copy["rgb"], device=self.device).unsqueeze(0)
        obs_copy["rgb"] = rearrange(obs_copy["rgb"], 'b h w c -> b c h w')

        obs_copy['camera_poses'] = torch.as_tensor(obs_copy['camera_poses'], device=self.device).unsqueeze(0)

        H, W = obs_copy['rgb'].shape[2:]
        sq_size = min(H, W)

        temp_rgb = obs_copy['rgb']

        temp_rgb = temp_rgb[:, :, :sq_size, :sq_size]
        temp_rgb = transforms.Resize((224, 224),
            interpolation = transforms.InterpolationMode.BILINEAR)(temp_rgb)

        obs_copy['rgb'] = temp_rgb
        temp_depth = torch.as_tensor(obs_copy['depth'], device=self.device).unsqueeze(0).unsqueeze(0)

        temp_depth = temp_depth[:, :sq_size, :sq_size]
        temp_depth = transforms.Resize((224, 224),
            interpolation = transforms.InterpolationMode.NEAREST)(temp_depth)
        temp_depth[np.isnan(temp_depth.cpu())] = 0
        obs_copy['depth'] = temp_depth

        obs_copy["state"] = torch.as_tensor(obs_copy["state"], device=self.device).unsqueeze(0)
        obs_copy['K_matrices'] = torch.as_tensor(obs_copy['K_matrices'], device=self.device).unsqueeze(0)

        del obs_copy["stamp"]
        return obs_copy

    def reset(self):
        ret = self._reset()
        while robot_utils.is_xarm_in_error():
            ret = self._reset()
            rospy.sleep(0.1)
        return ret

    def _reset(self):
        robot_utils.recover_xarm_from_error()
        if self.agent is not None:
            self.agent.reset()
        self._gripper_action_prev = 1  # open

        self._ri.ungrasp()
        rospy.sleep(1)

        self._robot.reset_pose()
        self._ri.angle_vector(self._robot.angle_vector(), time=5)
        self._ri.wait_interpolation()

        self._ri.grasp()
        obs = self.get_obs(rospy.Time.now(), is_first=True)
        self.vid = []

        print(f"reset!")
        return obs

    def step(self, action):
        control = action["control"]
        gripper_action = action["gripper"]

        if gripper_action.item() != self._gripper_action_prev:
            if gripper_action == 0:
                self._ri.ungrasp()
            elif gripper_action == 1:
                self._ri.grasp()
            self._gripper_action_prev = gripper_action


        curr_obs = self.get_obs(rospy.Time.now(), is_first=False)
        p_ee_in_link0 = curr_obs["state"]
        T_ee_in_link0 = self.control_to_target_coords(control, p_ee_in_link0)
        target_coords = skrobot.coordinates.Coordinates(
            pos=T_ee_in_link0[:3, 3], rot=T_ee_in_link0[:3, :3]
        )
        av = self._robot.rarm.inverse_kinematics(target_coords)

        ik_fail = av is False
        if ik_fail:
            raise RuntimeError("IK failed. reset")

        if not (self.dry or ik_fail):
            if not self.use_mouse:
                self._ri.angle_vector(self._robot.angle_vector(), time=.5)
                self._ri.wait_interpolation()
            else:
                self._ri.angle_vector(self._robot.angle_vector(), time=.5)
        return self.get_obs(rospy.Time.now(), is_first=False)

    def act(self, obs):
        if self.traj is not None:
            action = torch.as_tensor(next(self.traj), device=self.device).unsqueeze(0)
        elif self.agent is not None:
            with torch.no_grad():
                action = self.agent.act(obs)
        else:
            raise NotImplementedError

        act = {}
        act["control"] = action[:, :7]
        act["gripper"] = action[:, 7] > 0.5

        return act

    def control_to_target_coords(self, control, p_ee_in_link0):
        """
        control is delta position and delta quaternion.
        """
        p_ee_in_link0 = p_ee_in_link0.clone().to(control.device)
        p_ee_in_link0 = p_ee_in_link0.squeeze(0).cpu().numpy()
        control = control.squeeze(0).cpu().numpy()

        pos = p_ee_in_link0[:3]
        quat = p_ee_in_link0[3:]

        delta_pos = control[:3]

        if self.scaled_actions:
            delta_pos = delta_pos * 0.1
            delta_quat = control[3:]
        delta_quat = delta_quat / np.linalg.norm(delta_quat) # normalize quaternion to unit quaternion

        pose_ee = Pose(pos, quat)
        control_pose = Pose(delta_pos, delta_quat)

        if self.ee_frame_control:
            final_pose = pose_ee * control_pose
        else:
            final_pose = control_pose * pose_ee

        return final_pose.to_transformation_matrix()

    def save_vid(self):
        # save self.vid to wandb. it is a list of images
        vid = np.array(self.vid)

        if len(vid) <= 30:
            print('discarding trial because it is too short')
            return

        if self.interactive:
            k = input('(s)ave or (d)iscard trial: ')
        else:
            k = 's'

        if k == 's':
            vid = vid.transpose(0, 3, 1, 2)
            wandb.log({"video": wandb.Video(vid[::4], fps=30, format="mp4")}, step=self.num_iter)
            self.vid = []
            self.num_iter += 1
        else:
            print('discarding trial by user choice')

    def close(self):
        self.save_vid()
        if len(self.all_vid) > 30:
            all_vid = np.array(self.all_vid[::16])
            all_vid = all_vid.transpose(0, 3, 1, 2)
            wandb.log({"all_video": wandb.Video(all_vid, fps=30, format="mp4")}, step=self.num_iter)
            print(f"saved all video at {self.num_iter}")
        wandb.finish()
        self._ri.ungrasp()

@click.command()
@click.option("--config", "-c", type=str, default="agent config from agent training")
@click.option("--ckpt", "-a", type=str, default=None)
def main(config, ckpt):
    from simple_bc._interfaces.encoder import Encoder
    from simple_bc._interfaces.policy import Policy

    config = OmegaConf.load(config)
    
    assert ckpt is not None, "ckpt is required"
    encoder = Encoder.load_encoder(config.encoder)
    policy = Policy.load_policy(encoder, config.policy)
    policy.load_state_dict(torch.load(ckpt))

    agent = Agent(policy, )

if __name__ == "__main__":
    main()

#!/usr/bin/env python
import os
import sys
import time

import numpy as np
from gdict.data import GDict

import skrobot
import torch

import cv_bridge
import message_filters
import rospy
import tf
import tf.transformations as ttf

from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool

import time

from utils.conversion_utils import compute_forward_action, preproc_obs, Pose, to_torch
from utils import robot_utils
import click
import wandb

from omegaconf import OmegaConf

class Agent:
    def __init__(self,
                 policy,
                 scaled_actions,
                 ee_control):
        
        self.scaled_actions = scaled_actions
        self.ee_control = ee_control
        self.policy = policy

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
        self._obs = {}
        self._obs['rgb'] = []
        self._obs['depth'] = []
        self._obs['camera_poses'] = []
        self._obs['K_matrices'] = []
        self._obs['state'] = []
        
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

        # keep a running queue of self.policy.encoder.num_frames observations
        T = self.policy.encoder.num_frames
        self._obs['rgb'].append(rgb_wrist)
        self._obs['depth'].append(depth_wrist)
        self._obs['state'].append(p_ee_in_link0)
        self._obs['camera_poses'].append(T_camera_in_link0)
        self._obs['K_matrices'].append(K_wrist)

        self._obs['rgb'] = self._obs['rgb'][-T:]
        self._obs['depth'] = self._obs['depth'][-T:]
        self._obs['state'] = self._obs['state'][-T:]
        self._obs['camera_poses'] = self._obs['camera_poses'][-T:]
        self._obs['K_matrices'] = self._obs['K_matrices'][-T:]

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

    def get_obs(self):
        while not self._obs:
            rospy.sleep(0.001)
        assert self._obs is not None

        rgb = np.array(self._obs["rgb"]).transpose(0, 3, 1, 2)
        depth = np.array(self._obs["depth"])
        state = np.array(self._obs["state"])
        camera_poses = np.array(self._obs["camera_poses"])
        K_matrices = np.array(self._obs["K_matrices"])

        obs = preproc_obs(rgb, depth, camera_poses, K_matrices, state)

        for k in obs:
            obs[k] = to_torch(obs[k], device=self.policy.device).float().unsqueeze(0)

        return obs

    def reset(self):
        ret = self._reset()
        while robot_utils.is_xarm_in_error():
            ret = self._reset()
            rospy.sleep(0.1)
        return ret

    def _reset(self):
        robot_utils.recover_xarm_from_error()
        self._gripper_action_prev = 1  # open

        self._ri.ungrasp()
        rospy.sleep(1)

        self._robot.reset_pose()
        self._ri.angle_vector(self._robot.angle_vector(), time=5)
        self._ri.wait_interpolation()

        self._ri.grasp()
        obs = self.get_obs()
        self.vid = []

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


        curr_obs = self.get_obs()
        p_ee_in_link0 = curr_obs["state"][-1, -1, :] # last element of [B, frame, proprio]
        T_ee_in_link0 = self.control_to_target_coords(control, p_ee_in_link0)
        print(T_ee_in_link0)
        target_coords = skrobot.coordinates.Coordinates(
            pos=T_ee_in_link0[:3, 3], rot=T_ee_in_link0[:3, :3]
        )
        av = self._robot.rarm.inverse_kinematics(target_coords)

        ik_fail = av is False
        if ik_fail:
            raise RuntimeError("IK failed. reset")

        if not ik_fail:
            self._ri.angle_vector(self._robot.angle_vector(), time=.5)
        return self.get_obs()

    def act(self, obs):
        with torch.no_grad():
            action = self.policy(obs)

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
        pose_ee = Pose(*p_ee_in_link0)
        control_pose = Pose(*control)

        final_pose = compute_forward_action(pose_ee, control_pose, ee_control=self.ee_control, scaled_actions=self.scaled_actions)

        return final_pose.to_44_matrix()

    def save_vid(self):
        # save self.vid to wandb. it is a list of images
        vid = np.array(self.vid)

        if len(vid) <= 30:
            print('discarding trial because it is too short')
            return

        vid = vid.transpose(0, 3, 1, 2)
        wandb.log({"video": wandb.Video(vid[::4], fps=30, format="mp4")}, step=self.num_iter)
        self.vid = []
        self.num_iter += 1

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
@click.option("--train_config", "-c", type=str, default="agent config from agent training")
@click.option("--conv_config", "-d", type=str, default="dataset config from conversion")
@click.option("--pol_ckpt", "-a", type=str, default=None)
def main(train_config, conv_config, pol_ckpt):
    from simple_bc._interfaces.encoder import Encoder
    from simple_bc._interfaces.policy import Policy

    train_config = OmegaConf.load(train_config)
    assert pol_ckpt is not None, "ckpt is required"
    encoder = Encoder.load_encoder(train_config.encoder.value)
    policy = Policy.load_policy(encoder, train_config.policy.value)
    policy.load_state_dict(torch.load(pol_ckpt))
    policy.to(policy.device)

    conv_config = OmegaConf.load(conv_config)
    scaled_actions = conv_config.scaled_actions
    ee_control = conv_config.ee_control
    proprio = train_config.train_dataset.value.aug_cfg.use_proprio

    agent = Agent(policy, scaled_actions, ee_control)

    wandb.init(project="simple-bc", name="test")
    
    r = rospy.Rate(5)
    control_pub = rospy.Publisher("/control/status", Bool, queue_size=1)
    while True:
        obs = agent.reset()
        while not rospy.is_shutdown():
            try:
                action = agent.act(obs)
                obs = agent.step(action)
                control_pub.publish(True)
                r.sleep()
            except RuntimeError as e:
                agent.save_vid()
                break
        # input("Press enter to continue...")


if __name__ == "__main__":
    main()

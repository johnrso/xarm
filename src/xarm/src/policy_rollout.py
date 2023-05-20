#!/usr/bin/env python

import os
import sys
import time

import numpy as np
from gdict.data import GDict

np.set_printoptions(precision=3, suppress=True)
import skrobot
import torch
import datetime
import cv_bridge
import message_filters
import rospy
import tf
import tf.transformations as ttf
import threading

from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool

import time

from xarm_utils.conversion_utils import compute_forward_action, preproc_obs, Pose, to_torch, to_numpy
from xarm_utils import robot_utils
import wandb
import h5py
from pynput import keyboard

from simple_bc.utils.log_utils import pretty_print_cfg
from simple_bc.utils.visualization_utils import make_grid_video_from_numpy

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
np.set_printoptions(precision=3, suppress=True)
from omegaconf import OmegaConf

class Agent:
    def __init__(self,
                 encoder,
                 policy,
                 scale_factor,
                 ee_control,
                 rotation_mode,
                 proprio,
                 use_depth=True,
                 traj=None,
                 device="cuda"):

        self.scale_factor = scale_factor

        # TODO: fix later,
        self.ee_control = ee_control
        self.rotation_mode = rotation_mode
        self.encoder = encoder
        self.policy = policy
        self._ri = None
        self.device = device
        self.T = self.encoder.num_frames
        self.proprio = proprio
        self.use_depth = use_depth

        self.succ = None

        if traj is None:
            self.traj = None
            self.wandb = True
        else:
            with open(traj, "rb") as f:
                if traj is not None:
                    traj = h5py.File(traj, "r")
                    traj_id = list(traj.keys())[0]
                    self.traj = iter(traj[traj_id]["dict_str_actions"])
            self.wandb = True

        self.vid = []
        self.all_vid = []
        self.actions = []

        self.num_iter = 0

        # policy setup
        rospy.init_node("agent", anonymous=True)
        rospy.on_shutdown(self.close)

        self._setup_listeners()
        self._setup_robot()

    def _setup_robot(self):
        self._gripper_action_prev = True  # true if closed, false if open
        self._robot = robot_utils.XArm()
        self._ri = robot_utils.XArmROSRobotInterface(self._robot, namespace="")
        self._ri.update_robot_state()
        self._robot.angle_vector(self._ri.angle_vector())

    def _setup_listeners(self):
        self._obs = []
        self._current_obs = None

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
            queue_size=1,
        )
        sync.registerCallback(self._obs_callback)

        sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._sub_rgb_wrist,
                self._sub_rgb_base,
            ],
            slop=0.1,
            queue_size=1,
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

        position, quaternion = self._tf_listener.lookupTransform(
            target_frame="link_base",
            source_frame="link_tcp",
            time=rospy.Time(0),
        )

        # rotate from xyzw to wxyz
        quaternion = np.array(quaternion)[[3, 0, 1, 2]]
        p_ee_in_link0 = np.concatenate([position, quaternion]) # wxyz quaternion

        rgb_wrist = rgb_wrist.transpose(2, 0, 1) * 1.0
        rgb_base = rgb_base.transpose([2, 0, 1]) * 1.0

        obs = preproc_obs(rgb=rgb_wrist,
                          depth=depth_wrist,
                          rgb_base=rgb_base,
                          depth_base=depth_base,
                          camera_poses=T_camera_in_link0,
                          K_matrices=K_wrist,
                          state=p_ee_in_link0,
                          rotation_mode=self.rotation_mode)

        self._current_obs = GDict(obs).unsqueeze(0)

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

    def get_obs(self):
        while self._current_obs is None:
            rospy.sleep(0.001)

        curr_obs = self._current_obs

        self._obs += [curr_obs]
        while len(self._obs) < self.T:
            self._obs = [curr_obs] + self._obs
        if len(self._obs) > self.T:
            self._obs.pop(0)

        obs = GDict.stack(self._obs, axis=1).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
        return obs

    def reset(self):
        self.kill = False
        self.actions = []
        self.policy.reset()
        print('policy reset')
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
        self._ri.angle_vector(self._robot.angle_vector(), time=4)
        self._ri.wait_interpolation()

        self._obs = []
        self._current_obs = None

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

        p_ee_in_link0 = to_numpy(self._current_obs["state"][-1, :]) # last element of [B, frame, proprio]
        T_ee_in_link0 = self.control_to_target_coords(control, p_ee_in_link0)
        target_coords = skrobot.coordinates.Coordinates(
            pos=T_ee_in_link0[:3, 3], rot=T_ee_in_link0[:3, :3]
        )
        av = self._robot.rarm.inverse_kinematics(target_coords)

        ik_fail = av is False
        if ik_fail:
            raise RuntimeError("IK failed. reset")

        if not ik_fail:
            t = time.time()
            self._ri.angle_vector(self._robot.angle_vector(), time=.5)
        return self.get_obs()

    def act(self, obs):
        if self.traj is not None:
            action = next(self.traj)
        else:
            with torch.no_grad():
                if not self.proprio:
                    obs['state'] = torch.zeros_like(obs['state'])
                if not self.use_depth:
                    obs['depth'] = torch.zeros_like(obs['depth'])
                action = self.policy.act(self.encoder(obs))
                if len(action.shape) > 1:
                    action = action[0]
        self.actions.append(action)

        act = {}
        act["control"] = action[:7]
        act["gripper"] = action[7] > 0.5
        return act

    def control_to_target_coords(self, control, p_ee_in_link0):
        """
        control is delta position and delta quaternion.
        """
        control = control * self.scale_factor
        if self.rotation_mode == "quat":
            pose_ee = Pose.from_quaternion(*p_ee_in_link0)
            control_pose = Pose.from_quaternion(*control)
        elif self.rotation_mode == "aa":
            pose_ee = Pose.from_axis_angle(*p_ee_in_link0)
            control_pose = Pose.from_axis_angle(*control)
        elif self.rotation_mode == "euler":
            pose_ee = Pose.from_euler(*p_ee_in_link0)
            control_pose = Pose.from_euler(*control)
        else:
            raise NotImplementedError

        final_pose = compute_forward_action(pose_ee,
                                            control_pose,
                                            ee_control=self.ee_control)

        return final_pose.to_44_matrix()

    def save_vid(self):
        log_dict = {}
        # save self.vid to wandb. it is a list of images
        if len(self.actions) > 30:
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            actions = np.array(self.actions)

            for d in range(8):
                d1, d2 = d // 4, d % 4
                axs[d1, d2].plot(actions[:, d])
                axs[d1, d2].set_title(f"action {d}")

            # get this as an np array
            fig.canvas.draw()
            traj_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            traj_img = traj_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            log_dict["actions"] = wandb.Image(traj_img)

        vid = np.array(self.vid)

        if len(vid) <= 30:
            print('discarding trial because it is too short')
            return

        vid = vid.transpose(0, 3, 1, 2)
        if self.succ == 'y':
            success = 1
        elif self.succ == 'q':
            self.vid = []
            self.succ = None
            return
        else:
            success = 0

        self.succ = None
        log_dict["video"] = wandb.Video(vid[::4], fps=30, format="mp4")
        log_dict["success"] = success

        if self.wandb:
            wandb.log(log_dict, step=self.num_iter)

        self.vid = []
        self.all_vid.append(vid.transpose(0, 2, 3, 1))
        self.num_iter += 1

        return success

    def close(self):
        if self.wandb:
            all_vid = self.all_vid[:]
            make_grid_video_from_numpy(all_vid, 5, f"{wandb.run.dir}/all_vid.mp4", speedup=8)
            wandb.log({"all_vid": wandb.Video(f"{wandb.run.dir}/all_vid.mp4", fps=30, format="mp4")},
                      step=self.num_iter)
            print(f"saved video to {wandb.run.dir}/all_vid.mp4")
        if self._ri is not None:
            self._ri.ungrasp()
        if self.wandb:
            wandb.finish()

    def raise_error(self, key):
        self.kill = True
        self.succ = key

def main(train_config, conv_config, pol_ckpt, enc_ckpt, traj=None, tag=None):
    assert tag is not None, "tag is required"

    from simple_bc._interfaces.encoder import Encoder
    from simple_bc._interfaces.policy import Policy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = OmegaConf.load(train_config)

    pretty_print_cfg(train_config.encoder)
    pretty_print_cfg(train_config.policy)

    assert pol_ckpt is not None and enc_ckpt is not None, "ckpt is required"
    encoder = Encoder.build_encoder(train_config.encoder).to(device).eval()
    policy = Policy.build_policy(encoder.out_shape,
                                 train_config.policy,
                                 train_config.encoder).to(device).eval()  # Updated shape
    if policy.num_bins != -1:
        policy.load_kmeans_cluster(task=train_config.task, k=policy.num_bins)

    policy.load_state_dict(torch.load(pol_ckpt))
    encoder.load_state_dict(torch.load(enc_ckpt))

    conv_config = OmegaConf.load(conv_config)
    scale_factor = conv_config.scale_factor
    rotation_mode = conv_config.rotation_mode
    ee_control = conv_config.ee_control
    proprio = train_config.dataset.aug_cfg.use_proprio
    try:
        use_depth = train_config.dataset.aug_cfg.use_depth
    except:
        use_depth = True

    print(f"args to agent: scale_factor: {scale_factor}, \
          rotation_mode: {rotation_mode}, ee_control: {ee_control}, \
          proprio: {proprio}, use_depth: {use_depth}")

    if traj is None:
        save_dir = "/data/demo/"
        save_dir = os.path.join(save_dir, tag)
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(save_dir, exist_ok=True)

        wandb.init(project="internet-manipulation-test",
                   name=tag,
                   dir=save_dir)

    agent = Agent(encoder, policy, scale_factor, ee_control, rotation_mode, proprio, use_depth, traj)

    r = rospy.Rate(5)
    control_pub = rospy.Publisher("/control/status", Bool, queue_size=1)

    def on_press(key):
        try:
            if key.char in "qyn":
                agent.raise_error(key.char)
                return False
        except AttributeError:
            pass

    succ = []
    while True:
        obs = agent.reset()
        steps = 0
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print(f"begin trial; press q to discard, y for success, and anything else for failure")
        while not rospy.is_shutdown():
            try:
                action = agent.act(obs)
                r.sleep()
                obs = agent.step(action)
                control_pub.publish(True)
            except RuntimeError as e:
                print(e)
                break
            steps += 1
            if agent.kill:
                break
        val = agent.save_vid()
        if val is not None:
            succ.append(val)
        if len(succ) == 0:
            m = 0
        else:
            m = np.mean(succ)

        kill = input(f"\nkill? (y/n) succ rate = {m}, {len(succ)} trials thus far): ")
        if kill[-1] == 'y':
            print(f"\nsuccess rate: {m} ({len(succ)} trials)")
            print(f"media can be found at {save_dir}")
            rospy.signal_shutdown("done")
            exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None)
    parser.add_argument("--conv_config", type=str, default=None)
    parser.add_argument("--pol_ckpt", type=str, default=None)
    parser.add_argument("--enc_ckpt", type=str, default=None)
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))

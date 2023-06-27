#!/usr/bin/env python
import click
import sys
import threading

import time

import rospy
import numpy as np
import pyspacemouse
import skrobot

np.set_printoptions(precision=3, suppress=True)

import tf.transformations as ttf
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool
import pickle
import xarm_utils.robot_utils as robot_utils

@click.command()
@click.option('--rotation-mode', default='euler', help='Rotation mode: rpy or euler')
@click.option('--angle-scale', default=0.05, help='Angle scale')
@click.option('--translation-scale', default=0.02, help='Translation scale')
@click.option('--invert-control/--no-invert-control', is_flag=True, default=True, help='Invert control')
@click.option('--control-hz', default=5, help='Control frequency')
@click.option('--alpha', default=1., help='Alpha')
@click.option('--safe', is_flag=True, default=False, help='Safe mode limits the range of motion')
def main(rotation_mode, angle_scale, translation_scale, invert_control, control_hz, alpha, safe):
    kc = KeyboardControl(rotation_mode, angle_scale, translation_scale, invert_control, control_hz, alpha, safe)

class KeyboardControl:
    def __init__(self,
                rotation_mode: str = "euler",
                angle_scale: float = 0.1,
                translation_scale: float = 0.02,
                invert_control: bool = True,
                control_hz: float = 5,
                alpha: float = 0.5,
                safe: bool = False
            ):

        print('Rotation mode:', rotation_mode)
        print("Invert control:", invert_control)

        assert rotation_mode in ["rpy", "euler"], "rotation_mode must be rpy or euler; rpy is about own frame, euler is about world frame"
        assert min(angle_scale, translation_scale) > 0, "scale must be positive"

        rospy.init_node("keyboard_control", anonymous=True)
        rospy.loginfo("keyboard_control node started")
        self.state_lock = threading.Lock()
        self._pub = rospy.Publisher("/record_demo", Bool, queue_size=1)
        self._rotation_mode = rotation_mode
        invert = -1 if invert_control else 1
        self._angle_scale = angle_scale * invert
        self._translation_scale = translation_scale
        self._rpy_deadzone = 0.9 # the raw value must be greater than this to rotate the EE.
        self._robot = robot_utils.XArm()
        self.safe = safe

        self._control_pub = rospy.Publisher("/control/command", TransformStamped, queue_size=1)
        self._gripper_pub = rospy.Publisher("/control/gripper", Bool, queue_size=1)
        self.control_hz = rospy.Rate(control_hz)

        self._ri = robot_utils.XArmROSRobotInterface(self._robot, namespace='')
        self._ri.update_robot_state()
        self._gripper_state = "open"
        self._robot.angle_vector(self._ri.potentio_vector())

        self.next_mouse_state = None
        self.next_mouse_state_arr = []
        self.all_actions = []
        self._alpha = alpha # exponential smoothing parameter
        self.i = 0
        self.delta_button = 0
        success = pyspacemouse.open()
        if not success:
            print("Failed to open the spacemouse")
            sys.exit(1)

        # open read_spacemouse thread
        self.in_control = False
        self._thread = threading.Thread(target=self.read_spacemouse)
        self._thread.start()

        # register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)

        while not rospy.is_shutdown():
            key = input("o --> reset pose, p --> start control, q --> quit: ")
            if key == "p":
                self._pub.publish(True)
                print("starting control; left button --> gripper, right button --> save, both buttons --> discard")
                self.in_control = True
                self.control()
            elif key == "o":
                print("resetting robot pose")
                self.reset_robot_pose()
            elif key == "q":
                rospy.signal_shutdown("keyboard interrupt")
                sys.exit(0)
                break

    def control(self):
        assert self.in_control, "control is not started"
        T_ee_in_link0 = None
        self._gripper_state = "open"
        self.mouse_state_arr = np.zeros(6)
        while self.in_control:
            if T_ee_in_link0 is None:
                T_ee_in_link0 = (
                    self._robot.rarm.end_coords.worldcoords().T()
                )
                continue

            if self.next_mouse_state.buttons[0] == 1 and self.next_mouse_state.buttons[1] == 1:
                # end control
                print("ending control and discarding demo")
                self.in_control = False
                self.all_actions = []
                self._pub.publish(True)
                continue
            elif self.delta_button == 1 and self._gripper_state == "open":
                # close gripper
                self._ri.grasp()
                self._gripper_state = "closed"
                with self.state_lock:
                    self.delta_button = False
            elif self.delta_button == 1 and self._gripper_state == "closed":
                # open gripper
                self._ri.ungrasp()
                self._gripper_state = "open"
                with self.state_lock:
                    self.delta_button = False
            elif self.next_mouse_state.buttons[1] == 1:
                # end control
                print("ending control and saving demo")

                # import matplotlib.pyplot as plt
                # T = len(self.all_actions)
                # # 6 subfigures in 1x6. plot each dimension of self.all_actions
                # fig, axs = plt.subplots(1, 6, figsize=(20, 5))
                # for i in range(6):
                #     axs[i].plot(np.arange(T), np.array(self.all_actions)[:, i])
                #     axs[i].set_title(f"dim {i}")
                # plt.savefig(f"demo_{self.i}_alpha_{self._alpha}.png")

                # print(f"saved demo_{self.i}_alpha_{self._alpha}.png")
                self.i += 1
                self.all_actions = []
                self.in_control = False
                self._pub.publish(False)
                self.reset_robot_pose()

                continue

            self.mouse_state_arr = self._alpha * self.next_mouse_state_arr + (1 - self._alpha) * self.mouse_state_arr
            # Suppress small values on other dimensions when control is high in one dimension

            if np.max(np.abs(self.mouse_state_arr)) > 0.1:
                self.mouse_state_arr[np.abs(self.mouse_state_arr) < 0.05] = 0
            # Add noise to the mouse state
            self.mouse_state_arr += np.random.normal(0, 0.02, 6)
            self.all_actions.append(self.mouse_state_arr)
            tx, ty, tz, r, p, y = self.mouse_state_arr

            ee_pos = T_ee_in_link0[:3, 3]
            ee_rot = T_ee_in_link0[:3, :3]

            trans_transform = np.eye(4)
            trans_transform[:3, 3] = np.array([ty,-tx,tz]) * self._translation_scale

            # break rot_transform into each axis
            rot_transform_x = np.eye(4)
            rot_transform_x[:3, :3] = ttf.quaternion_matrix(
                ttf.quaternion_from_euler(-r * self._angle_scale, 0, 0)
            )[:3, :3]

            rot_transform_y = np.eye(4)
            rot_transform_y[:3, :3] = ttf.quaternion_matrix(
                ttf.quaternion_from_euler(0, -p * self._angle_scale, 0)
            )[:3, :3]

            rot_transform_z = np.eye(4)
            rot_transform_z[:3, :3] = ttf.quaternion_matrix(
                ttf.quaternion_from_euler(0, 0, y * self._angle_scale)
            )[:3, :3]

            rot_transform = rot_transform_x @ rot_transform_y @ rot_transform_z

            new_ee_pos = trans_transform[:3, 3] + ee_pos
            if self._rotation_mode == "rpy":
                new_ee_rot = ee_rot @ rot_transform[:3, :3]
            elif self._rotation_mode == "euler":
                new_ee_rot = rot_transform[:3, :3] @ ee_rot

            if self.safe and new_ee_pos[2] < 0.05:
                new_ee_pos[2] = 0.05

            T_ee_in_link0[:3, 3] = new_ee_pos
            T_ee_in_link0[:3, :3] = new_ee_rot

            self.move_to_pose(new_ee_pos, new_ee_rot)
            self.publish_control(T_ee_in_link0)
            self.control_hz.sleep()

    def move_to_pose(self, new_ee_pos, new_ee_rot, wait_interp=False, time=0.5):
        self._robot.rarm.inverse_kinematics(
            skrobot.coordinates.Coordinates(
                pos=new_ee_pos, rot=new_ee_rot
            )
        )

        self._ri.angle_vector(self._robot.angle_vector(), time=time)

        if wait_interp:
            self._ri.wait_interpolation()

    def publish_control(self, T_ee_in_link0):
        # convert to a quaternion
        q = ttf.quaternion_from_matrix(T_ee_in_link0)

        # publish T_ee_in_link0
        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = "link0"
        transform_msg.child_frame_id = "ee"
        transform_msg.transform.translation.x = T_ee_in_link0[0, 3]
        transform_msg.transform.translation.y = T_ee_in_link0[1, 3]
        transform_msg.transform.translation.z = T_ee_in_link0[2, 3]
        transform_msg.transform.rotation.x = q[0]
        transform_msg.transform.rotation.y = q[1]
        transform_msg.transform.rotation.z = q[2]
        transform_msg.transform.rotation.w = q[3]

        self._control_pub.publish(transform_msg)
        self._gripper_pub.publish(self._gripper_state == "closed") # true if closed, false if open

    def reset_robot_pose(self):
        robot_utils.recover_xarm_from_error()
        if self._gripper_state == "closed":
            self._ri.ungrasp()
            rospy.sleep(1)
            self._gripper_state = "open"

        self._robot.reset_pose()
        self._ri.angle_vector(self._robot.angle_vector(), time=4)
        self._ri.wait_interpolation()
        self.delta_button = 0

    def read_spacemouse(self):
        while 1:
            if self.next_mouse_state is None:
                ms = pyspacemouse.read()
                self.next_mouse_state = ms
                self.next_mouse_state_arr = np.array([ms.x, ms.y, ms.z, ms.roll, ms.pitch, ms.yaw])
                continue
            curr_button = self.next_mouse_state.buttons[0]
            ms = pyspacemouse.read()
            self.next_mouse_state = ms
            self.next_mouse_state_arr = np.array([ms.x, ms.y, ms.z, ms.roll, ms.pitch, ms.yaw])

            if not self.delta_button:
                with self.state_lock:
                    self.delta_button = max(self.next_mouse_state.buttons[0] - curr_button, 0)

    def shutdown_hook(self):
        self._thread.join()
        pyspacemouse.close()
        self._pub.publish(False)
        self._ri.ungrasp()

if __name__ == "__main__":
    main()
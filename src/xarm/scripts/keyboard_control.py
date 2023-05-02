#!/usr/bin/env python
import click
import sys
import threading

import time

import rospy
import numpy as np
import pyspacemouse
import skrobot

import tf.transformations as ttf
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool
import pickle
import xarm_utils.robot_utils as robot_utils

@click.command()
@click.option('--rotation-mode', default='euler', help='Rotation mode: rpy or euler')
@click.option('--angle-scale', default=0.1, help='Angle scale')
@click.option('--translation-scale', default=0.04, help='Translation scale')
@click.option('--invert-control/--no-invert-control', is_flag=True, default=True, help='Invert control')
@click.option('--control-hz', default=5, help='Control frequency')
def main(rotation_mode, angle_scale, translation_scale, invert_control, control_hz):
    kc = KeyboardControl(rotation_mode, angle_scale, translation_scale, invert_control, control_hz)

class KeyboardControl:
    def __init__(self,
                rotation_mode: str = "euler",
                angle_scale: float = 0.1,
                translation_scale: float = 0.02,
                invert_control: bool = True,
                control_hz: float = 5
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

        self._control_pub = rospy.Publisher("/control/command", TransformStamped, queue_size=1)
        self._gripper_pub = rospy.Publisher("/control/gripper", Bool, queue_size=1)
        self.control_hz = rospy.Rate(control_hz)

        self._ri = robot_utils.XArmROSRobotInterface(self._robot, namespace='')
        self._ri.update_robot_state()
        self._gripper_state = "closed"
        self._robot.angle_vector(self._ri.potentio_vector())

        self.mouse_state = None
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
        while self.in_control:
            if T_ee_in_link0 is None:
                T_ee_in_link0 = (
                    self._robot.rarm.end_coords.worldcoords().T()
                )
                continue

            if self.mouse_state.buttons[0] == 1 and self.mouse_state.buttons[1] == 1:
                # end control
                print("ending control and discarding demo")
                self.in_control = False
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
            elif self.mouse_state.buttons[1] == 1:
                # end control
                print("ending control and saving demo")
                self.in_control = False
                self._pub.publish(False)
                self.reset_robot_pose()

                continue

            ee_pos = T_ee_in_link0[:3, 3]
            ee_rot = T_ee_in_link0[:3, :3]

            trans_transform = np.eye(4)
            trans_transform[:3, 3] = np.array([self.mouse_state.y,
                                               -self.mouse_state.x,
                                               self.mouse_state.z]) * self._translation_scale

            r, p, y = self.mouse_state.roll, self.mouse_state.pitch, self.mouse_state.yaw


            def abs_scale(x):
                sgn = np.sign(x)
                abs_x = abs(x)

                return sgn * (abs_x - self._rpy_deadzone) / (1 - self._rpy_deadzone)

            r = 0 if abs(r) < self._rpy_deadzone else abs_scale(r)
            p = 0 if abs(p) < self._rpy_deadzone else abs_scale(p)
            y = 0 if abs(y) < self._rpy_deadzone else abs_scale(y)

            # break rot_transform into each axis
            rot_transform_x = np.eye(4)
            rot_transform_x[:3, :3] = ttf.quaternion_matrix(
                ttf.quaternion_from_euler(-self.mouse_state.roll * self._angle_scale, 0, 0)
            )[:3, :3]

            rot_transform_y = np.eye(4)
            rot_transform_y[:3, :3] = ttf.quaternion_matrix(
                ttf.quaternion_from_euler(0, -self.mouse_state.pitch * self._angle_scale, 0)
            )[:3, :3]

            rot_transform_z = np.eye(4)
            rot_transform_z[:3, :3] = ttf.quaternion_matrix(
                ttf.quaternion_from_euler(0, 0, self.mouse_state.yaw * self._angle_scale)
            )[:3, :3]

            rot_transform = rot_transform_x @ rot_transform_y @ rot_transform_z

            new_ee_pos = trans_transform[:3, 3] + ee_pos
            if self._rotation_mode == "rpy":
                new_ee_rot = ee_rot @ rot_transform[:3, :3]
            elif self._rotation_mode == "euler":
                new_ee_rot = rot_transform[:3, :3] @ ee_rot

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
        self._ri.wait_interpolation() # suitable angle vector to allow nicer random initializations

        T_ee_link_0 = (self._robot.rarm.end_coords.worldcoords().T())
        
        reset_pos = T_ee_link_0[:3, 3] + np.random.uniform(low = [0.0, -0.05, -0.1], high = [0.1, 0.05, -0.0])

        reset_angle, reset_axis, _ = ttf.rotation_from_matrix(T_ee_link_0)
        reset_angle += np.random.uniform(low = -np.pi/12, high = np.pi/12)
        reset_axis += np.random.uniform(low = -0.005, high = 0.005, size=3)
        reset_rot = ttf.rotation_matrix(reset_angle, reset_axis)[:3, :3]
        
        self.move_to_pose(reset_pos, reset_rot, wait_interp=True, time=1)

        self.delta_button = 0

        # if self._gripper_state == "open":
        #     self._ri.grasp()
        #     rospy.sleep(1)
        #     self._gripper_state = "closed"

    def read_spacemouse(self):
        while 1:
            if self.mouse_state is None:
                self.mouse_state = pyspacemouse.read()
                continue
            curr_button = self.mouse_state.buttons[0]
            self.mouse_state = pyspacemouse.read()
            if not self.delta_button:
                with self.state_lock:
                    self.delta_button = max(self.mouse_state.buttons[0] - curr_button, 0)

    def shutdown_hook(self):
        self._thread.join()
        pyspacemouse.close()
        self._pub.publish(False)
        self._ri.ungrasp()

if __name__ == "__main__":
    main()
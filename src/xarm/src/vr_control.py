#!/usr/bin/env python

import argparse
import copy
import datetime
import pickle
# import sys
import sys
import time

import numpy as np
import openvr
import path
from termcolor import cprint
from xarm_msgs.msg import RobotMsg
from std_msgs.msg import Bool

import cv_bridge
import message_filters
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
import skrobot
import tf
import tf.transformations as ttf

import utilities as utils


class XArmControlInterface:
    def __init__(self, controller_id: int, real: bool = False):
        self._controller_id = controller_id
        self._real = real

        # if self._real:
        #     while not utils.recover_panda_from_error():
        #         rospy.sleep(1)

        # skrobot
        self._robot = utils.XArm()
        if real:
            self._ri = utils.XArmROSRobotInterface(self._robot, namespace='')
            self._ri.update_robot_state()
            # self._ri.ungrasp()
            self._gripper_state = "open"
            self._robot.angle_vector(self._ri.potentio_vector())
        else:
            self._viewer = skrobot.viewers.TrimeshSceneViewer()
            self._viewer.add(self._robot)
            self._viewer.show()

        # openvr
        openvr.init(openvr.VRApplication_Scene)
        self._vrsystem = openvr.VRSystem()
        self.wait_for_detection()

        # ui
        self._button_clicked = {
            "trackpad_pressed_button": False,
            "trigger": False,
        }

        # demo recording
        self._control_started_at = None
        self._recording_occurring = False
        self._control_ended_at = None
        self._recorded_dir = None
        self._robot2 = copy.deepcopy(self._robot)
        self._tf_listener = tf.listener.TransformListener(
            cache_time=rospy.Duration(30)
        )

        self.first_record = True
        self._sub_xarm_state = message_filters.Subscriber(
            "/xarm/xarm_states", RobotMsg
        )
        sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._sub_xarm_state,
            ],
            slop=0.1,
            queue_size=50,
        )
        sync.registerCallback(self._xarm_state_callback)
        # create a publisher to send a boolean to /record_demo topic
        self._pub_record_demo = rospy.Publisher("/record_demo", Bool, queue_size=1)

    def __del__(self):
        openvr.shutdown()

    def wait_for_detection(self):
        cprint(
            f"==> Waiting for detecting the controller: {self._controller_id}"
        )
        detected = False
        while not detected:
            result, _ = self._vrsystem.getControllerState(self._controller_id)
            detected = result == 1
        cprint(f"==> Detected the controller: {self._controller_id}")

    def run(self):
        # IR: commented these two lines (don't seem right)
        # self._button_clicked['trackpad_pressed_button'] = True
        # self.control(True)
        while True:
            # print('line 100, before getting')
            result, state = self._vrsystem.getControllerState(
                self._controller_id
            )

            # print('line 104, after getting state')
            assert result == 1, f"expected 1 but got {result}"
            state = utils.dict_from_controller_state(state)
            if state["menu_button"]:
                cprint("==> Starting to control the robot (restrict z axis)")
                self._button_clicked["trackpad_pressed_button"] = True
                self.control(True)
            elif state["trackpad_pressed_button"] == "none":
               self._button_clicked["trackpad_pressed_button"] = False
            elif self._button_clicked["trackpad_pressed_button"]:
                pass
            elif state["trackpad_pressed_button"] == "bottom":
                cprint("==> Starting to control the robot")
                self._button_clicked["trackpad_pressed_button"] = True
                self.control()
            elif state["trackpad_pressed_button"] == "top":
                cprint("==> Quiting the application")
                self._button_clicked["trackpad_pressed_button"] = True
                self.reset_robot_pose()
                break
            elif state["trackpad_pressed_button"] == "center":
                print("YO")
                if not self._recording_occurring:
                    cprint("==> Starting the demo recording")
                    self._button_clicked["trackpad_pressed_button"] = True
                    self._pub_record_demo.publish(True)
                    self._recording_occurring = True
                else:
                    cprint("==> Ignoring choice - recording already in progress")
            elif state["trackpad_pressed_button"] == "left":
                if self._recording_occurring:
                    cprint("==> Saving the demo recording")
                    self._button_clicked["trackpad_pressed_button"] = True
                    #self.save_demo_recording()
                    self._pub_record_demo.publish(False)
                    self._recording_occurring = False
                    self.reset_robot_pose()
                else:
                    cprint("==> Ignoring choice - no recording in progress")
            elif state["trackpad_pressed_button"] == "right":
                if self._recording_occurring:
                    cprint("==> Discarding the demo recording")
                    self._button_clicked["trackpad_pressed_button"] = True
                    #   self.discard_demo_recording()
                    self._pub_record_demo.publish(True)
                    self._recording_occurring = False
                    self.reset_robot_pose()
                else:
                    cprint("==> Ignoring choice - no recording in progress")
            else:
                pass


    def save_demo_recording(self):
        if self._recorded_dir is None:
            cprint("==> Demo recording is not running", color="yellow")
        else:
            home = path.Path("~").expanduser()
            saved_dir = (
                home
                / ".ros/arm_xarm/1017_turn_faucet"
                / self._recorded_dir.basename()
            )

            root_saved_dir = home / ".ros/arm_xarm/1017_turn_faucet"
            saved_dir.parent.makedirs_p()
            self._recorded_dir.move(saved_dir)
            num_dirs = len([name for name in root_saved_dir.listdir() if name.isdir()])
            cprint(f"==> Saved demo recording to: {saved_dir}; contains {num_dirs} demos", color="green")
            # get the total number of directories in saved_dir

            self._recorded_dir = None
        self.reset_robot_pose()

    def discard_demo_recording(self):
        if self._recorded_dir is None:
            cprint("==> Demo recording is not running", color="yellow")
        else:
            try:
                self._recorded_dir.rmtree_p()
            except OSError:
                pass
            cprint(
                f"==> Discarded demo recording: {self._recorded_dir}",
                color="yellow",
            )
            self._recorded_dir = None
        self.reset_robot_pose()

    def _xarm_state_callback(
        self, state_msg: RobotMsg
    ):
        self._in_error = state_msg.err != 0

    def reset_robot_pose(self):
        if self._in_error:
            utils.recover_xarm_from_error()
        if self._gripper_state == "closed":
            # self._robot.pre_reset_pose()
            # self._ri.angle_vector(self._robot.angle_vector())
            # self._ri.wait_interpolation()
            self._ri.ungrasp()
            rospy.sleep(1)
            self._gripper_state = "open"

            # self._robot.pre_reset_pose()
            # self._ri.angle_vector(self._robot.angle_vector(), time_scale=10)
            # self._ri.wait_interpolation()

        # ee_coords = self._robot.rarm.end_coords.copy_worldcoords()
        # ee_coords.translate([0, 0.0, 0.3], wrt="world")
        # T_ee_in_link0 = ee_coords.T()
        # self._robot.rarm.inverse_kinematics(
        #     skrobot.coordinates.Coordinates(
        #         pos=T_ee_in_link0[:3, 3], rot=T_ee_in_link0[:3, :3]
        #     )
        # )
        # self._ri.angle_vector(self._robot.angle_vector(), time=1)
        # self._ri.wait_interpolation()

        self._robot.reset_pose()
        self._ri.angle_vector(self._robot.angle_vector(), time_scale=10)
        self._ri.wait_interpolation()

    def control(self, restrict_z=False):
        self._control_started_at = rospy.Time.now()
        # restrict_z = True
        T_ee_t0_in_link0 = None
        T_controller_t0_in_base = None
        while True:
            if self._in_error:
                utils.recover_xarm_from_error()
                T_ee_t0_in_link0 = None
                T_controller_t0_in_base = None

            pose = self._vrsystem.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, 16
            )
            pose = pose[self._controller_id].mDeviceToAbsoluteTracking
            T_controller_in_base = utils.matrix_from_openvr_pose(pose)

            result, state = self._vrsystem.getControllerState(
                self._controller_id
            )
            assert result == 1
            state = utils.dict_from_controller_state(state)

            # print(state)

            if state["trackpad_pressed_button"] == "none":
                self._button_clicked["trackpad_pressed_button"] = False
            elif (
                state["trackpad_pressed_button"] == "bottom"
                and not self._button_clicked["trackpad_pressed_button"]
            ):
                cprint("==> Stopping to control the robot")
                self._button_clicked["trackpad_pressed_button"] = True
                break


            if self._real:
                if state["trigger"] == 0:
                    self._button_clicked["trigger"] = False
                elif (
                    state["trigger"] > 0.5
                    and not self._button_clicked["trigger"]
                ):
                    if self._gripper_state == "open":
                        cprint("==> Closing the gripper")
                        self._ri.grasp()
                        rospy.sleep(0.5)
                        self._gripper_state = "closed"
                        # restrict_z = True
                        T_controller_t0_in_base = None
                    elif self._gripper_state == "closed":
                        cprint("==> Opening the gripper")
                        self._ri.ungrasp()
                        rospy.sleep(0.5)
                        self._gripper_state = "open"
                        restrict_z = False
                        T_controller_t0_in_base = None
                    else:
                        raise ValueError
                    self._button_clicked["trigger"] = True

            # Map the XYZ of controller to the XYZ of the robot base
            T_controller_in_base = (
                T_controller_in_base
                @ ttf.rotation_matrix(np.deg2rad(90), [0, 1, 0])
                @ ttf.rotation_matrix(np.deg2rad(90), [1, 0, 0])
            )

            if T_controller_t0_in_base is None:
                T_controller_t0_in_base = T_controller_in_base
                T_ee_t0_in_link0 = (
                    self._robot.rarm.end_coords.worldcoords().T()
                )
                continue

            # if restrict_z:
            #     T_controller_in_base[2:3, 3] = T_controller_t0_in_base[2:3, 3]
            #     T_controller_in_base[:3, :3] = T_controller_t0_in_base[:3, :3]

            T_base_in_controller_t0 = np.linalg.inv(T_controller_t0_in_base)
            T_controller_in_controller_t0 = (
                T_base_in_controller_t0 @ T_controller_in_base
            )

            T_controller_in_controller_t0 = utils.scale_transformation(
                T_controller_in_controller_t0, scale=1.0
                # T_controller_in_controller_t0, scale=0.2
            )

            T_ee_in_ee_t0 = T_controller_in_controller_t0
            T_ee_in_link0 = T_ee_t0_in_link0 @ T_ee_in_ee_t0

            if restrict_z:
                T_ee_in_link0[2:3, 3] = T_ee_t0_in_link0[2:3, 3]
                T_ee_in_link0[:3, :3] = T_ee_t0_in_link0[:3, :3]

            # T_ee_in_link0[:3, :3] = T_ee_t0_in_link0[:3, :3]  # TODO: TEMP: dont usee rotation

            # position = ttf.translation_from_matrix(T_ee_in_link0)
            # print(position)

            self._robot.rarm.inverse_kinematics(
                skrobot.coordinates.Coordinates(
                    pos=T_ee_in_link0[:3, 3], rot=T_ee_in_link0[:3, :3]
                )
            )
            if not self._real:
                self._viewer.redraw()

            if self._real:
                # TMP DEBUG
                self._ri.angle_vector(self._robot.angle_vector(), time=1)

        self._ri.wait_interpolation()

        self._control_ended_at = rospy.Time.now()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--controller_id", type=int, default=1, help="controller id"
    )
    args = parser.parse_args()

    rospy.init_node("control_xarm")

    interface = XArmControlInterface(
        controller_id=args.controller_id, real=True
    )
    interface.run()

if __name__ == "__main__":
    main()
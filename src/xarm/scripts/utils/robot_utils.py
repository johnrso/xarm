import tempfile
import os
import subprocess

import actionlib
import control_msgs
import numpy as np
import rospy
import skrobot
import xarm_gripper.msg
from skrobot.interfaces.ros.base import ROSRobotInterfaceBase
from skrobot.models.urdf import RobotModelFromURDF

class XArmROSRobotInterface(ROSRobotInterfaceBase):
    WIDTH_MAX = 800
    SPEED_MAX = 3000

    def __init__(self, *args, **kwargs):
        super(XArmROSRobotInterface, self).__init__(*args, **kwargs)

        self.gripper_move = actionlib.SimpleActionClient(
            '/xarm/gripper_move',
            xarm_gripper.msg.MoveAction)
        self.gripper_move.wait_for_server()

    @property
    def rarm_controller(self):
        return dict(
            controller_type='rarm_controller',
            controller_action='xarm/xarm7_traj_controller/follow_joint_trajectory',  # NOQA
            controller_state='xarm/xarm7_traj_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[j.name for j in self.robot.rarm.joint_list],
        )

    def default_controller(self):
        return [self.rarm_controller]

    def grasp(self, target_pulse=0, **kwargs):
        self.move_gripper(target_pulse=target_pulse, **kwargs)

    def ungrasp(self, **kwargs):
        self.move_gripper(target_pulse=self.WIDTH_MAX, **kwargs)

    def move_gripper(self, target_pulse, pulse_speed=SPEED_MAX, wait=True):
        goal = xarm_gripper.msg.MoveGoal(target_pulse=target_pulse, pulse_speed=pulse_speed)
        if wait:
            # self.gripper_move.send_goal_and_wait(goal)
            self.gripper_move.send_goal(goal)
            rospy.sleep(0.2)
        else:
            self.gripper_move.send_goal(goal)

class XArm(RobotModelFromURDF):
    def __init__(self, *args, **kwargs):
        urdf = rospy.get_param("/robot_description")
        tmp_file = tempfile.mktemp()
        with open(tmp_file, "w") as f:
            f.write(urdf)
        super().__init__(urdf_file=tmp_file)
        os.remove(tmp_file)
        self.reset_pose()

    @property
    def rarm(self):
        link_names = ["link{}".format(i) for i in range(1, 8)]
        links = [getattr(self, n) for n in link_names]
        joints = [link.joint for link in links]
        model = skrobot.model.RobotModel(link_list=links, joint_list=joints)
        model.end_coords = skrobot.coordinates.CascadedCoords(
            parent=self.link_tcp,
            name="rarm_end_coords",
        )
        return model

    def pre_reset_pose(self):
        av = [
            1.47, 0.42, -2.0, 1.75, 0.45, 1.9, -0.4
        ]
        self.rarm.angle_vector(av)
        return self.angle_vector()

    def reset_pose(self):
        av = [
            0, -1.22, 0, 1.404, 0 , 1.80, 0 # bag pose
        ]
        av = np.array(av) + np.random.uniform(-0.025, 0.025)
        self.rarm.angle_vector(av)
        return self.angle_vector()

def recover_xarm_from_error():
    # subprocess.call(["rosservice", "call", "/xarm/clear_err"],
    #                 stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
    # subprocess.call(["rosservice", "call", "/xarm/set_mode", "1"],
    #                 stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
    subprocess.call(["rosservice", "call", "/xarm/moveit_clear_err"],
                    stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)


def is_xarm_in_error():
    res = subprocess.Popen(["rostopic", "echo", "-n", "1", "/xarm/xarm_states"],
                           stdout=subprocess.PIPE).communicate()[0]
    res = res.decode("utf-8")
    return 'err: 0' not in res
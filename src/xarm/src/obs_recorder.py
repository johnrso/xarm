#!/usr/bin/env python
import os
from pathlib import Path
import shutil
import xarm_utils.robot_utils as robot_utils
import click

import rospy
import numpy as np
import message_filters

import datetime
import pickle

import tf
import tf.transformations as ttf
from cv_bridge import CvBridge

from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, JointState, Image

@click.command()
@click.option('--demo-dir', default='/data/demo', help='Demo directory')
@click.option('--record-act/--no-record-act', default=True, help='Record actions')
def main(demo_dir, record_act):
    ObsRecorder(demo_dir, record_act)

class ObsRecorder:
    def __init__(self, demo_dir: str = "/data/demo", record_act=True):
        rospy.init_node("obs_recorder", anonymous=True)

        # check if the demo_dir exists. if not, create it
        self._demo_dir = Path(demo_dir)
        self._demo_dir.mkdir(parents=True, exist_ok=True)
        self.len_traj = 0

        self._control_started_at = None
        self._control_ended_at = None
        self._recorded_dir = None
        self._robot = robot_utils.XArm()
        self._gripper_state = None
        self._tf_listener = tf.listener.TransformListener(
            cache_time=rospy.Duration(30)
        )

        self.record_flag_sub = rospy.Subscriber("/record_demo", Bool, self.record_flag_callback)

        # wrist camera
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

        self._sub_control = message_filters.Subscriber(
            "/control/command", TransformStamped
        )

        self._sub_gripper = message_filters.Subscriber(
            "/control/gripper", Bool
        )

        self.recording_pub = rospy.Publisher("/record_demo/status", Bool, queue_size=1)
        self.record_act = record_act

        if not self.record_act:
            import threading
            self.pub_control = rospy.Publisher("/control/command", TransformStamped, queue_size=1)
            self.pub_gripper = rospy.Publisher("/control/gripper", Bool, queue_size=1)

            self._pub_thread = threading.Thread(target=self._pub_thread_fn)
            self._pub_thread.start()

        sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._sub_caminfo_wrist,
                self._sub_rgb_wrist,
                self._sub_depth_wrist,
                self._sub_caminfo_base,
                self._sub_rgb_base,
                self._sub_depth_base,
                self._sub_joint,
                self._sub_control,
                self._sub_gripper,
            ],
            slop=0.1,
            queue_size=50,
            allow_headerless=True,
        )
        sync.registerCallback(self._demo_recording_callback)
        rospy.loginfo("obs_recorder: node started")

        log_file = self._demo_dir / "log.txt"
        if not log_file.exists():
            log_file.touch()
        log_msg = input("describe the demo set being collected. this will be written to the log file.\n")
        log_msg = str(datetime.datetime.now()) + " " + log_msg
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        while True:
            inp = input("d --> delete last recording\n")
            if inp == "d":
                try:
                    demo_dirs = sorted(os.listdir(self._demo_dir))
                    demo_dirs = [d for d in demo_dirs if d not in ["log.txt", "_conv"]]
                    to_remove = self._demo_dir / demo_dirs[-1]

                    import shutil
                    shutil.rmtree(to_remove, ignore_errors=True)
                    print("Removed", to_remove)
                except:
                    print("no recordings to remove.")

    def _pub_thread_fn(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rospy.loginfo_once("obs_recorder: publishing dummy control")
            control_msg = TransformStamped()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.header.frame_id = "world"
            control_msg.child_frame_id = "world"
            control_msg.transform.translation.x = 0
            control_msg.transform.translation.y = 0
            control_msg.transform.translation.z = 0
            control_msg.transform.rotation.x = 0
            control_msg.transform.rotation.y = 0
            control_msg.transform.rotation.z = 0
            control_msg.transform.rotation.w = 1

            gripper_msg = Bool()
            gripper_msg.data = False

            self.pub_control.publish(control_msg)
            self.pub_gripper.publish(gripper_msg)
            rate.sleep()

    def _demo_recording_callback(
        self,
        caminfo_msg_wrist,
        rgb_msg_wrist,
        depth_msg_wrist,
        caminfo_msg_base,
        rgb_msg_base,
        depth_msg_base,
        joint_msg,
        control_msg,
        gripper_msg
    ):
        rospy.loginfo_once("obs_recorder: messages synced")
        stamp = caminfo_msg_wrist.header.stamp
        if (
            self._control_started_at is None
            or self._recorded_dir is None
            or stamp < self._control_started_at
            or (
                self._control_ended_at is not None
                and self._control_ended_at > self._control_started_at
                and stamp > self._control_ended_at
            )
        ):
            return
        bridge = CvBridge()

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

        # proprio processing
        joint_positions = np.array(joint_msg.position)
        joint_velocites = np.array(joint_msg.velocity)

        # Need in pose form
        position, quaternion = self._tf_listener.lookupTransform(
            target_frame="link_base",
            source_frame="link_tcp",
            time=rospy.Time(0),
        )

        # quaternion is in xyzw form. convert to axis angle

        quaternion = np.array(quaternion)[[3, 0, 1, 2]]
        p_ee_in_link0 = np.concatenate([position, quaternion])

        # Notes:
        # T_cam_to_world = T_camera_in_link0: MUST go from camera to world frame transform
        # K_wrist matrix will be recorded in K_wrist form and then inverted during preprocessing by model
        position, quaternion = self._tf_listener.lookupTransform(
            target_frame="link_base",
            source_frame="wrist_camera_color_optical_frame",
            time=rospy.Time(0),
        )

        T_camera_in_link0 = ttf.quaternion_matrix(quaternion)
        T_camera_in_link0[:3, 3] = position

        p, q = control_msg.transform.translation, control_msg.transform.rotation

        position = [p.x, p.y, p.z]
        quaternion = [q.w, q.x, q.y, q.z]
        control = np.concatenate([position, quaternion])

        dt = datetime.datetime.fromtimestamp(stamp.to_sec())
        try:
            if self._recorded_dir and self._recorded_dir.exists():
                recorded_file = self._recorded_dir / (dt.isoformat() + ".pkl")
                with open(recorded_file, "wb") as f:
                    pickle.dump(
                        dict(
                            timestamp=dt.isoformat(),
                            rgb_wrist=rgb_wrist,
                            depth_wrist=depth_wrist,
                            K_wrist=K_wrist,
                            rgb_base=rgb_base,
                            depth_base=depth_base,
                            K_base=K_base,
                            p_ee_in_link0=p_ee_in_link0,
                            T_camera_in_link0=T_camera_in_link0,
                            joint_positions=joint_positions,
                            joint_velocites=joint_velocites,
                            gripper_state=gripper_msg.data, # true if closed, false if open
                            control=control,
                        ),
                        f
                    )
                self.len_traj += 1
                print('\r', "traj len: ", str(self.len_traj), end = '')
        except Exception as e:
            pass
        self.recording_pub.publish(True)

    def record_flag_callback(self, msg):
        # get the number of demos in self._demo_dir
        if msg.data and self._control_started_at is None:
            rospy.loginfo("obs_recorder: recording started")
            rospy.loginfo("Folder currently have {} demos".format(len(os.listdir(self._demo_dir))))
            self.len_traj = 0
            self._recorded_dir = self._demo_dir / datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self._recorded_dir.mkdir(parents=True, exist_ok=True)
            self._control_started_at = rospy.Time.now()
            self._control_ended_at = None
        elif msg.data:
            # delete the current demo
            if self._recorded_dir is not None:
                shutil.rmtree(self._recorded_dir, ignore_errors=True)
            self._recorded_dir = None
            num_demos = len(os.listdir(self._demo_dir))
            self._control_ended_at = rospy.Time.now()
            self._control_started_at = None
            rospy.loginfo_once("obs_recorder: record_flag received")
            rospy.loginfo("obs_recorder: recording deleted: num demos = %d" % num_demos)
            rospy.loginfo("Remaining demos: {}".format(os.listdir(self._demo_dir)))
        elif not msg.data and self._control_started_at is not None:
            self._control_ended_at = rospy.Time.now()
            self._control_started_at = None
            self._recorded_dir = None
            num_demos = len(os.listdir(self._demo_dir))
            rospy.loginfo_once("obs_recorder: record_flag received")
            rospy.loginfo("obs_recorder: recording saved: num demos = %d" % num_demos)
            rospy.loginfo("Current demo: {}".format(os.listdir(self._demo_dir)))
        else:
            rospy.loginfo("obs_recorder: recording flag ignored")

    # on shutdown, check if recording a demo. If so, save the demo
    def shutdown(self):
        if self._control_started_at is not None:
            self._control_ended_at = rospy.Time.now()
            self._control_started_at = None
            rospy.loginfo("obs_recorder: recording ended")

if __name__ == "__main__":
    main()

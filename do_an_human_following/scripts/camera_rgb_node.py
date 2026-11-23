#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_rgb_node.py - ROS Node độc lập xuất bản topic /camera/image_raw qua GStreamer
"""
import os
import sys
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.camera_streamer import build_gstreamer_pipeline


def main():
    rospy.init_node("camera_rgb_node", anonymous=False)
    pub = rospy.Publisher(config.ROS_IMAGE_TOPIC, Image, queue_size=1)
    bridge = CvBridge()

    pipeline = build_gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        rospy.logwarn("[CameraNode] Không mở được CSI GStreamer, fallback sang USB camera.")
        cap = cv2.VideoCapture(config.SENSOR_ID)

    rate = rospy.Rate(config.CAMERA_FPS)
    rospy.loginfo(f"[CameraNode] Đang phát topic {config.ROS_IMAGE_TOPIC} ở {config.CAMERA_FPS} FPS")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret and frame is not None:
            msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "camera_rgb_link"
            pub.publish(msg)
        rate.sleep()

    cap.release()


if __name__ == "__main__":
    main()
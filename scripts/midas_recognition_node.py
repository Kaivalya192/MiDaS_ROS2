#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np

bridge = CvBridge()

class MidasRecognitionNode(Node):

    def __init__(self):
        super().__init__('midas_recognition')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10)
        self.subscription 

        self.img_pub = self.create_publisher(Image, "/midas_result_image", 1)

        # Load MiDaS model
        self.model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type, pretrained=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

    def camera_callback(self, data):
        img = bridge.imgmsg_to_cv2(data, "bgr8")

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply MIDAS transformation
        input_batch = self.transform(img_rgb).to(self.device)

        # Perform inference
        with torch.no_grad():
            prediction = self.midas(input_batch)

            # Resize prediction to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert prediction to numpy array
        depth_image = prediction.cpu().numpy()

        # Normalize depth image to range [0, 255]
        depth_image_normalized = ((depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)

        # Publish depth image
        depth_img_msg = bridge.cv2_to_imgmsg(depth_image_normalized)
        self.img_pub.publish(depth_img_msg)


def main(args=None):
    rclpy.init(args=args)
    midas_recognition_node = MidasRecognitionNode()
    rclpy.spin(midas_recognition_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

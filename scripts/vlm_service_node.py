#!/home/user/pel_ws/pel_venv/bin/python

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from orvis.srv import VLM, VLMResponse

import requests
import base64
import cv2
from cv_bridge import CvBridge
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
bridge = CvBridge()

VLM_MODELS = [
    "llama-3.2-90b-vision-preview"
]

# Encode cv2 image to base64 string
def encode_image_cv2(cv_image):
    _, buffer = cv2.imencode(".jpg", cv_image)
    return base64.b64encode(buffer).decode("utf-8")

def make_handler(model):
    def handler(req):
        try:
            cv_image = bridge.imgmsg_to_cv2(req.image, "bgr8")
            image_b64 = encode_image_cv2(cv_image)

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": req.prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                "temperature": 0.3
            }

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                     headers=headers, json=payload)

            if response.status_code != 200:
                return VLMResponse(success=False, message=response.text)

            result = response.json()["choices"][0]["message"]["content"]
            return VLMResponse(success=True, message=result)

        except Exception as e:
            return VLMResponse(success=False, message=str(e))
    return handler

def main():
    rospy.init_node("vlm_service_node")

    for model in VLM_MODELS:
        srv_name = f"/vlm_query_{model.replace('-', '_').replace('.', '_')}"
        rospy.Service(srv_name, VLM, make_handler(model))
        rospy.loginfo(f"Service started: {srv_name}")

    rospy.spin()

if __name__ == "__main__":
    main()

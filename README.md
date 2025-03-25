# Ontology-Based Robotic Vision System (ORVIS)


This repository contains the ROS package for Ontology-based Robotic VIsion System (ORVIS), that is designed to interface with different computer vision models available on the [HuggingFace](https://huggingface.co/) platform.

The system uses an ontology to reason about which models are suitable for a given task and to semantically interpret their outputs. It supports perception-driven knowledge acquisition and grounding.

The current implementation targets ROS Noetic and assumes a robot or simulator that publishes RGB images to a camera topic.

## How to get started

Clone the repo in the ROS workspace of your choice, then run

```cd <your_catkin_workspace>
cd <your_catkin_workspace>/src
git clone https://github.com/Dorteel/orvis.git
cd ../
pip install -r src/orvis/requirements.txt
catkin_make
source devel/setup.bash
```

## Description
ORVIS consists of the following main components:

* Model Manager: Loads and manages models from HuggingFace based on configuration files. Each model is exposed as a ROS service.

* Ontology Interface: Uses the ORKA ontology to reason over available models, their capabilities, and the perceived environment.

* Service Layer: Provides standardized ROS services for object detection, segmentation, and material recognition.

* Pipeline Node: Subscribes to camera data and orchestrates the execution of appropriate models based on high-level queries (e.g., "Find all apples on the table").

Models are described via YAML config files that specify their inputs, outputs, and capabilities (e.g., detects: apple or segments: material).

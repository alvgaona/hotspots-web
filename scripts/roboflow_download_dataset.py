#!/usr/bin/python3

import roboflow

api_key = "X3b0jaeYasLLGJfc8nNX"

rf = roboflow.Roboflow(api_key=api_key)

WORKSPACE_NAME = "solarpanelimages"
PROJECT_NAME = "solar-panel-infrared-images"
DATASET_VERSION = 5

project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
version = project.version(DATASET_VERSION)

IMAGE_ANNOTATION_FORMAT = "yolov11"

dataset = version.download(IMAGE_ANNOTATION_FORMAT)


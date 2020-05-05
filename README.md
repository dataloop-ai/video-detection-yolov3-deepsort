# Dataloop Function as a Service for Video Annotation and Tracking

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

This is a Yolov3 object detection with DeepSort tracking.
Can be used locally to detect a video.
Can be used as a function for the Dataloop platform.

---

## Run locally

1. clone this repo. This repo contains two open source repos:
    ```
    https://github.com/nwojke/deep_sort
    https://github.com/qqwweee/keras-yolo3
    ```
2. Create virtual env

3. Install requirements pacakges
    
    ```
    pip install -r requirements.txt
    ```
4. Download the weights (COCO pretrained and tracker encoder) and move to "model_data":
        
    ```
    https://storage.googleapis.com/dtlpy/model_assets/yolo-coco/coco_classes.txt
    https://storage.googleapis.com/dtlpy/model_assets/yolo-coco/yolo_anchors.txt
    https://storage.googleapis.com/dtlpy/model_assets/yolo-coco/yolo.h5
    https://storage.googleapis.com/dtlpy/model_assets/deepsort/mars-small128.pb
    ```
5. Run on a local video
    
    ```
    python detect_video.py <video filepath>
    ```

### Create and run Dataloop Function
1. Log in to platform and create a project
2. Push the package to the project
3. Upload artifacts 
4. Deploy a service
5. Run the remote Function
6. Optional - Create a trigger to execute the function on Dataloop events
# Object Tracking

# Installation

It is recommended to use a virtual environment. To install the dependencies, run:
```bash
pip install -r requirements.txt
```

The code was tested with Python 3.12.

You will also need to download the `ADL-Rundle-6` dataset and place it in the root of the project for parts 2 and 3, and inside the `Appearance_Aware_IOU_TP4` folder for part 4.

# Handouts

Every part (1-4) is done in a separate folder. All the code for each part is contained in its folder.
Execute the code to obtain each video:

## Part 1: 2D Kalman Filter

To run the 2D Kalman Filter tracking demo, execute the following command:

```bash
python 2D_Kalman-Filter_TP1/objTracking.py
```
A window will open showing the tracking of a ball. The video file is included in the project.

## Part 2: IOU Tracker

To run the IOU tracker, first make sure the `ADL-Rundle-6` dataset is in the root of the project. Then, run the following command:

```bash
python IOU_Tracker_TP2/iou_tracker.py
```
This will generate a text file with the tracking results and a video in the `IOU_Tracker_TP2/output` folder.

## Part 3: Kalman Guided IOU Tracker

To run the Kalman Guided IOU tracker, first make sure the `ADL-Rundle-6` dataset is in the root of the project. Then, run the following command:

```bash
python Kalman_Guided_IOU_TP3/kalman_iou_tracker.py
```
This will generate a text file with the tracking results and a video in the `Kalman_Guided_IOU_TP3/output` folder.

## Part 4: Appearance-Aware IOU Tracker

To run the Appearance-Aware IOU tracker, first make sure the `ADL-Rundle-6` dataset is inside the `Appearance_Aware_IOU_TP4` folder. Then, run the following command:

```bash
python Appearance_Aware_IOU_TP4/appearance_aware_iou_tracker.py
```
This will generate a text file with the tracking results and a video in the `Appearance_Aware_IOU_TP4/output` folder.

You can find the final video already in **output/ADL-Rundle-6.mp4** (part 5)

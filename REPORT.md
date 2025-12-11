# Project Report: Multi-Object Video Tracking

This report details the work undertaken to develop a multi-object tracker, progressing from a simple IOU-based tracker to a more sophisticated version incorporating a Kalman filter and an appearance-based Re-Identification (ReID) model. The project was divided into four main parts, building upon each other.

## Part 1 & 2: Foundations - Kalman Filters and IOU Tracking

The first two parts of the project involved understanding and utilizing existing codebases for single-object tracking and multi-object tracking.

### Tasks Undertaken:
- **Part 1 (Kalman Filter):** The provided code in `2D_Kalman-Filter_TP1/` demonstrated single-object tracking using a Kalman filter to predict and update the position of an object based on its centroid. This formed the basis for motion prediction in later stages.
- **Part 2 (IOU Tracker):** The code in `IOU_Tracker_TP2/` implemented a baseline multi-object tracker. This tracker used the Intersection over Union (IoU) metric to associate detections across frames, with the Hungarian algorithm finding the optimal assignment. This code was the starting point for our custom tracker.

## Part 3: Kalman-Guided IOU Tracker

The goal of this part was to improve the IOU tracker's motion prediction capabilities by integrating the Kalman filter.

### Tasks Undertaken:
1.  **Integration:** A new tracker was developed in the `Kalman_Guided_IOU_TP3/` directory by merging the logic from the IOU tracker and the Kalman filter.
2.  **State Prediction:** For each tracked object, a Kalman filter instance was created. Before associating detections in a new frame, the filter was used to predict the new position of each track.
3.  **Association:** The association cost matrix was calculated based on the IoU between the *predicted* bounding boxes and the new detections, making the association more robust to short-term occlusions.
4.  **State Update:** For successfully matched tracks, the Kalman filter's state was updated with the centroid of the new detection.

### Challenges Encountered and Solutions:
-   **Challenge:** The Kalman filter requires a precise time step (`dt`) between frames for its state-space model. This value was not hardcoded.
-   **Solution:** The `seqinfo.ini` file for the image sequence contained the frame rate. I implemented a function using Python's `configparser` library to read this file, extract the frame rate, and calculate the correct `dt` (1/frameRate).
-   **Challenge:** An initial error in the detection loading logic caused the script to fail. The code was attempting to parse space-separated data as if it were comma-separated.
-   **Solution:** The `load_detections` function was corrected to split the lines from the detection file by spaces instead of commas, aligning it with the actual file format.

## Part 4: Appearance-Aware IoU-Kalman Tracker

This was the final and most complex part, where an appearance-based Re-Identification model was added to the tracker to handle longer occlusions and improve identity preservation.

### Tasks Undertaken:
1.  **Feature Extraction:** A `FeatureExtractor` class was implemented to interface with the provided ONNX ReID model (`reid_osnet_x025_market1501.onnx`). This class handles the preprocessing of image patches (cropping, resizing, and normalization) and runs inference to generate a unique feature vector for each detected object.
2.  **Combined Cost Matrix:** The core of the tracker's association logic was modified. The cost for associating a track with a detection is now a weighted sum of two metrics:
    -   The geometric IoU distance (as in Part 3).
    -   The appearance distance (cosine distance between the track's feature vector and the detection's feature vector).
3.  **Track Feature Management:** Each track now maintains its own feature vector. When a track is matched, its feature vector is updated using an exponential moving average. This creates a more stable and robust appearance representation over time.

### Challenges Encountered and Solutions:
-   **Challenge:** The script initially failed due to a missing `onnxruntime` dependency required to run the ReID model.
-   **Solution:** A virtual environment was created for the project to manage dependencies cleanly. The `onnxruntime` library, along with other necessary packages like `numpy`, `scipy`, and `opencv-python`, was installed into this environment using `pip`.
-   **Challenge:** A persistent `KeyError` indicated that `configparser` was failing to read the `[Sequence]` section from the `seqinfo.ini` file, despite the file path being correct.
-   **Solution:** After debugging revealed that `configparser.read()` was not parsing the file correctly, the implementation was changed to manually open and read the file's content into a string, which was then successfully parsed using `config.read_string()`.
-   **Challenge:** The script later failed with a `FileNotFoundError`. The issue was traced to how relative paths were being resolved when the script was executed from the project's root directory by the agent.
-   **Solution:** All file paths within the script (`img_dir`, `det_file`, `seq_info_file`, `model_path`) were adjusted to be correctly relative to the project root, ensuring that all data files were found regardless of the execution context.
-   **Challenge:** The ONNX model threw a data type error, expecting `float32` input but receiving `float64`.
-   **Solution:** This was traced to the normalization step where numpy arrays defaulted to `float64`. The `dtype` of the mean and standard deviation arrays used for image normalization was explicitly set to `np.float32`, resolving the type mismatch.

## Conclusion

This project successfully progressed from utilizing basic tracking components to building and refining a sophisticated multi-object tracker. By systematically integrating a Kalman filter for motion prediction and an appearance-based ReID model, the tracker's ability to handle complex scenarios like object occlusion and re-identification was significantly improved. The challenges encountered were primarily related to environment setup, data handling, and library-specific issues, all of which were resolved through systematic debugging and code refinement.
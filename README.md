# Face Filter App

This project is a face filter application that applies various accessories like sunglasses, mustaches, and custom overlays using facial landmark detection. The application uses OpenCV, dlib, and Tkinter for image processing, facial recognition, and GUI interaction.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Code Overview](#code-overview)
    1. [Facial Accessories](#facial-accessories)
    2. [Frame Processor](#frame-processor)
    3. [GUI](#gui)
    4. [Main](#main)
    5. [Pose Estimation](#pose-estimation)
    6. [Testing](#testing)

## Installation

To run this project, you'll need to have Python installed along with several dependencies. Follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/augustgresens/Face-App.git
    ```

2. **Install dependencies**:

    ```bash
    pip install opencv-python
    pip install dlib
    pip install Pillow
    ```

3. **Prepare the shape predictor**:

   Download the `shape_predictor_68_face_landmarks.dat` file from dlib's models and place it in the `data_files` directory.


## Usage

To run the face filter application, use the following command:

```bash
python main.py
```

## Code Overview

### Facial Accessories
`facial_accessories.py` contains the `FacialAccessories` class, which manages the application of facial accessories.

#### Key Functions: 
1. `alpha_blend`
    - **Description**
        - Blends an overlay image with transparency onto a background frame.
    - **Key Features**
        - Uses alpha blending to integrate an image that has transparent parts onto a frame.
2. `compute_homography`
    - **Description**
        - Computes a homography transformation matrix between source points and the destination points.  
    - **Key Features**
        - Aligns an overlay with the face using perspective transformation.
        - Finds the homography transformation using `cv2.findHomography`, which is a method for geometric mapping between two images for perpective correction and image alignment.
3. `transform_overlay`
    - **Description**
        - Transforms an overlay using a given transformation matrix.
    - **Key Features**
        - Applies the homography transformation to the overlay image.
        - Uses cv2.warpPerspective for perspective transformation.
4. `add_mustache`
    - **Description**
        - Adds a mustache overlay to a frame based on facial landmarks.
    - **Key Features**
        - Utilizes `cv2.resize` to adjust the mustache to the appropriate size.
        - Uses `cv2.Rodrigues` to convert rotation vector into a rotation matrix, facilitating the calculation of 3D to 2D projection points, and properly adjust the pitch.
        - `cv2.projectPoints` converts 3D points to their corresponding 2D points
        - Uses rotation and translation vectors for accurate placement on face with pose.
        - Uses `compute_homography` to correctly align the mustache with where it should be, given correct source points and destination points.
        - Applies mustache with alpha blending
5. `add_sunglasses`
    - **Description**
        - Adds a sunglasses overlay to a frame based on facial landmarks.
    - **Key Features**
        - Utilizes `cv2.resize` to adjust the mustache to the appropriate size.
        - - Uses `cv2.Rodrigues` to convert rotation vector into a rotation matrix, facilitating the calculation of 3D to 2D projection points, and properly adjust the pitch.
        - `cv2.projectPoints` converts 3D points to their corresponding 2D points
        - Uses rotation and translation vectors for accurate placement on face with pose.
        - Uses `compute_homography` to correctly align the sunglasses with where it should be, given correct source points and destination points.
        - Applies sunglasses with alpha blending
6. `apply_overlay`
    - **Description**
        - Applies a custom overlay to a frame based on facial landmarks.
    - **Key Features**
        - Utilizes `cv2.resize` to adjust the mustache to the appropriate size.
        - `cv2.projectPoints` converts 3D points to their corresponding 2D points
        - Uses rotation and translation vectors for accurate placement on face with pose.
        - Uses `compute_homography` to correctly align the sunglasses with where it should be, given correct source points and destination points.
        - Applies sunglasses with alpha blending

### Frame Processor
`frame_processor.py` contains the FrameProcessor class, which processes video frames with facial landmark detection and applies various overlays.

#### Key Functions:
1. `process_frame`
    - **Description**
        - Processes a video frame to apply overlays.
    - **Key Features**
        - `cv2.cvtColor`: Converts the frame to grayscale.
        - `cv2.calcOpticalFlowPyrLK`: Compares feature points between two consecutive frames and computes their motion, and uses the Lucas-Kanade method for better precision in tracking movements. 
        - Updates current landmarks with newly detected landmarks, and if there is no detection, reverts to last good detection using `handle_last_good_detection`.
        - Uses NumPy arrays for matrix and vector operations.

2. `preprocess_frame`
    - **Description**
        - Preprocesses the frame by converting to graycale, and applying filters to help out with processing.
    - **Key Features**
        - `cv2.cvtColor`: Converts the frame to grayscale.
        - Uses adaptive histogram equalization through CLAHE for enhancing contrast which helps deal with some lighting issues.
        - `cv2.bilateralFilter`: Applies a bilateral smoothing filter to the grayscale frame for noise reduction while preserving edges. It replaces the intensity of each pixel with an average of intesity values from nearby filters to help deal with lighting issues.
        - `cv2.morphologyEx`: Improves image quality by eroding away boundaries in a binary image to reduce noise, and then expands the boundaries of objects by filling in occlusion gaps.
3. `detect_pose`
    - **Description**
        - Detects the pose and facial landmarks in a frame.
    - **Key Features**
        - `estimate_pose`: Estimate pose function uses `cv2.solvePNP` to solve the Perspective-n-Point problem to estimate object pose, and uses 3D and 2D points correspondences to estimate the pose of the head.
4. `handle_last_good_detection`
    - **Description** 
        - Handles cases where detection fails using the last known good detection.
    - **Key Features**
        - Persists the current detection for 10 frames in the case where the current detection fails.
        - Clears the frame history when a successful detection occurs.
5. `apply_overlays`
    - **Description**
        -  Applies overlays to the frame.
    - **Key Features**
        - `cv2.warpPerspective`: Applies a perspective transformation to an image, allowing adjustments in orientation and viewpoint to it's transformation matrices.
        - `cv2.addWeighted`: Blends two images together using specified weights.
        - Uses a camera matrix and distortion coefficients for accurate projection.
6. `get_camera_matrix`
    - **Description**
        - Returns the camera matrix. 
    - **Key Features**
        - Uses the pinhole camera model with the focal length equalling the width of the frame. 
        - Uses a NumPy array for storing the camera matrix.

### GUI
`gui.py` contains the GUI class for setting up a GUI using Tkinter, with buttons to toggle features.

#### Key Functions:
1. `setup_gui`
    - **Description**
        - Sets up the main GUI layout.
    - **Key Features**
        - Label for displaying the video feed and buttons to toggle different overlays.

2. `update_image`
    - **Description**
        - Updates the label with a new image.
    - **Key Features**
        - Converts an OpenCV image to a format that Tkinter can use, and updates the displayed images for Tkinter's use as well. 

3. `mainloop`
    - **Description**
        - Starts the main loop of the GUI.
    - **Key Features**
        - Keeps the Tkinter GUI active and responsive to events.
        
### Main
`main.py` contains the FaceApp class for the main face filter application.

#### Key Functions:
1. `update_flags`
    - **Description**
        - Updates the overlay flags based on input from the user.
    - **Key Features**
        - Allows users to toggle different facial accessories on or off, or clear overlays. 
        
2. `show_frame`
    - **Description**
        - Captures and displays a video frame.
    - **Key Features**
        - Reads a frame through the camera, processes it through `FrameProcessor`, and updates the GUI with the processed frame. 
        - Converts the frame to RGB using `cv2.cvtColor` before displaying to GUI.
        - Uses `self.root.after` to schedule the next frame which creates a loop. 
        

### Pose Estimation
`pose_estimation.py` contains the function estimate_pose for estimating the pose of a face in a video frame.
1. `estimate_pose`
    - **Description**
        - Estimates the pose of a face using the dlib detector and predictor for facial landmarks, and OpenCV's `solvePnP` function.
    - **Key Features**
        - Determines the 3D orientation of the face, and returns the rotation vector and the translation vector to describe it's position.
        - `cv2.cvtColor` converts the frame to grayscale if it's in BGR format.
        - `cv2.solvePnP` gives a solution to the Perspective-n-Point problem to estimate the pose using an observed 2D image and their corresponding 3D coordinates.
        - Uses the iterative method `cv2.SOLVEPNP_ITERATIVE` for pose etimation which balances computational efficiency and precision.

### Testing
`testing.py` contains unit tests for the facial filters using the unittest module.
1. `setUp`
    - **Description**
        - Initializes testing environment by loading samples and mock data.
2. `test_with_null_image_input`
    - **Description**
        - Tests how the `add_mustache` function handles a `None` input for the frame.
    - **Key Features**
        - The function should return `None`, which means that it gracefully handles misinputs.
3. `test_no_faces_detected`
    - **Description**
        - Evaluates response of the `add_mustache` function when no facial landmarks are detected.
    - **Key Features**
        - Uses a `MagicMock` to simulate a return value of `None` from the `compute_homography` function. 
        - The output image should remain unchanged.
4. `test_add_mustache`
    - **Description**
        - Verifies if the `add_mustache` function applies a mustache overlay properly given proper inputs.
    - **Key Features**
        - The resulting image should differ from the original image, indicating that the mustache has been applied.

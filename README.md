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
    git clone <repository-url>
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
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
`facial_accessories.py` contains the FacialAccessories class, which manages the application of facial accessories.

#### Key Functions: 
1. `alpha_blend`: Blends an overlay image with transparency onto a background frame.
2. `compute_homography`: Computes a homography transformation matrix.
3. `transform_overlay`: Transforms an overlay using a given transformation matrix.
4. `add_mustache`: Adds a mustache overlay to a frame based on facial landmarks.
5. `add_sunglasses`: Adds a sunglasses overlay to a frame based on facial landmarks.
6. `apply_overlay`: Applies a custom overlay to a frame based on facial landmarks.

### Frame Processor
`frame_processor.py` contains the FrameProcessor class, which processes video frames with facial landmark detection and applies various overlays.

#### Key Functions:
1. `process_frame`: Processes a video frame to apply overlays.
2. `preprocess_frame`: Preprocesses the frame.
3. `detect_pose`: Detects the pose and facial landmarks in a frame.
4. `handle_last_good_detection`: Handles cases where detection fails using the last known good detection.
5. `apply_overlays`: Applies overlays to the frame.
6. `get_camera_matrix`: Returns the camera matrix.

### GUI
`gui.py` contains the GUI class for setting up a GUI using Tkinter, with buttons to toggle features.

#### Key Functions:
1. `setup_gui`: Sets up the main GUI layout.
2. `update_image`: Updates the label with a new image.
3. `mainloop`: Starts the main loop of the GUI.

### Main
`main.py` contains the FaceApp class for the main face filter application.

#### Key Functions:
1. `update_flags`: Updates the overlay flags.
2. `show_frame`: Captures and displays a video frame.

### Pose Estimation
`pose_estimation.py` contains the function estimate_pose for estimating the pose of a face in a video frame.

### Testing
`testing.py` contains unit tests for the facial filters using the unittest module.

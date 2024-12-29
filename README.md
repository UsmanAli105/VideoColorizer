# Video Colorizer

## Overview

The **Video Colorizer** application allows users to upload black-and-white video files, process them for colorization, and download the colorized output. This project is divided into two parts:
1. **Frontend**: A simple web interface for uploading, processing, and downloading videos.
2. **Backend**: A Flask-based API that handles the actual video colorization using deep learning models.

## Features
- **Frontend:**
  - Drag and drop video upload interface.
  - Handles only video files (prevents uploading invalid files).
  - Processes the uploaded video on the server.
  - Allows users to download the processed video once it's ready.
  
- **Backend:**
  - Upload black-and-white video files.
  - Apply video colorization using ECCV and SIGGRAPH deep learning models.
  - Download the processed colorized video.
  - Cross-Origin Resource Sharing (CORS) enabled for easy integration.

## Getting Started

### Demo
<iframe width="560" height="315" src="https://www.youtube.com/embed/c3d-3vZSHp8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### Prerequisites
- A server running at `http://localhost:5000` to handle the file upload, processing, and download functionality.
- A modern web browser (Chrome, Firefox, etc.).
- Python 3.8+ installed on the server for backend setup.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/video-colorization.git
   cd video-colorization
   ```

2. **Frontend Installation:**
   - Open `index.html` in a web browser for the frontend interface.

3. **Backend Installation:**
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

   - Change Directory:
     ```bash
     cd Back-end
     ```


   - Create necessary directories:
     ```bash
     mkdir uploads output weights
     ```

   - Start the Flask application:
     ```bash
     python app.py
     ```

### Download Weights

To download the weights for this project, click the link below and select the desired files:

[Download Weights from Google Drive](https://drive.google.com/drive/folders/1Yoo1Zlt6w08o2rahgttjzptOhKnpy1-q?usp=sharing)


### Usage

1. **Frontend Usage:**
   - Open `index.html` in a browser.
   - Drag and drop a black-and-white video file into the drop area to upload.
   - Once the video is uploaded successfully, the **Process** and **Cancel** buttons will become enabled.
   - Click **Process** to start processing the video.
   - Once processing is complete, the **Download** button will be enabled.
   - Click **Download** to download the processed colorized video.

2. **Backend API Usage:**
   - **Upload a video:**
     ```bash
     curl -X POST -F "file=@path_to_video.mp4" http://127.0.0.1:5000/upload
     ```

   - **Process the uploaded video:**
     ```bash
     curl -X POST http://127.0.0.1:5000/process
     ```

   - **Download the colorized video:**
     ```bash
     curl -X POST http://127.0.0.1:5000/download --output output.mp4
     ```

## File Structure

```
/frontend
  ├── index.html            # Main HTML file
  ├── /css/style.css        # Custom CSS for styling the page
  └── /js/app.js            # JavaScript to handle video uploading, processing, and downloading

/backend
  ├── app.py                # Main Flask application
  ├── color.py              # Core video processing and colorization logic
  ├── dataset.py            # Custom dataset for video frame loading
  ├── models.py             # Deep learning models for colorization
  ├── utils.py              # Utility functions
  ├── uploads/              # Directory for uploaded videos
  ├── output/               # Directory for processed videos
  ├── weights/              # Directory for model weights
  ├── requirements.txt      # Python dependencies
  └── README.md             # Project documentation
```

## Requirements

- Python 3.8+
- Torch and TorchVision
- Flask
- Flask-CORS
- OpenCV
- scikit-image
- tqdm
- imageio

Install all dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Model Information

This project uses two pre-trained models:
- **ECCV Generator**: For colorizing videos.
- **SIGGRAPH Generator**: For enhancing the colorization quality.

These models are loaded from the `weights` directory during runtime.

## Notes

- Ensure the weights are correctly placed in the `weights` folder.
- The default output video is saved as `output/output.mp4`.
- Adjust `max_frames` and `fps` in `color.py` if needed for performance.

## Troubleshooting

- **Invalid file type error:** Ensure that you're uploading a valid video file (e.g., MP4, AVI, etc.).
- **Failed to upload:** Check that the server is running and accessible at `http://localhost:5000`.
- **Failed to process:** Ensure the models are correctly placed in the `weights` directory and that the server has sufficient resources to process the video.
```

This markdown combines the frontend and backend instructions into one cohesive document, covering both the user-facing interface and the backend logic behind the **Video Colorizer** application.
# YOLOv8 Live Webcam Object Detector

This project implements a **real-time object detection** application based on the **YOLOv8n** model, specifically designed to run within the constrained **Google Colab/Jupyter Notebook** environment. It leverages custom **JavaScript** and **ipywidgets** to access the user's local webcam and create a functional, non-blinking, low-latency "live" object detection feed.

Since Colab runs on a remote server, the standard local procedure (using `cv2.VideoCapture(0)`) cannot be used. This solution continuously captures a single, high-quality frame from the browser, processes that frame with YOLOv8, and updates an embedded image widget in-place.

---

## Key Features

* **Model:** **YOLOv8n** for fast and efficient object detection inference.
* **Environment:** Specifically tailored for **Google Colab/Jupyter Notebooks**.
* **Webcam Access:** Utilizes **JavaScript injection** to access the user's local camera through the browser.
* **Stable Output:** Employs **ipywidgets** to update the image in-place, effectively avoiding the common "blinking" effect experienced with standard real-time loops in Colab.
* **Status Indicators:** Provides clear status updates (capturing, processing, and displaying) during operation.

---

## Setup and Installation

### Exercise: Requirements

This project depends on the necessary libraries, which are typically listed in a `requirements.txt` file (e.g., `ultralytics`, `ipywidgets`, `opencv-python`, `Pillow`, etc.).

### 1. Install Dependencies

To install the required libraries before executing the main script, run the following command in a Colab code cell:

```bash
!pip install -r requirements.txt
```
### 2. Run app.py code in Google colab after installing dependencies.

### 3. ScreenShots are uploaded in output folder.

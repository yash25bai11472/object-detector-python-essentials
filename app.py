import cv2
from ultralytics import YOLO
import numpy as np
import base64
import time
from IPython import display
from ipywidgets import Button, Layout, VBox, Image, Label, HTML
from io import BytesIO
from google.colab import output
import sys

# initialize
print("Loading YOLOv8n model...")
try:
    # YOLO model
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have run 'pip install ultralytics opencv-python'.")
    sys.exit()


def get_webcam_frame(quality=0.8):
    """
    Captures a frame from the user's webcam using JavaScript and returns it as a
    NumPy array for OpenCV processing.
    """
    js = display.Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const video = document.createElement('video');
            video.style.display = 'block';

            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;

            await new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    resolve();
                }
            });

            await new Promise(resolve => setTimeout(resolve, 300));

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
        ''')
    display.display(js)
    data = output.eval_js('takePhoto({})'.format(quality))

    binary_data = base64.b64decode(data.split(',')[1])
    img_array = np.frombuffer(binary_data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame

# live camera detection function:
def run_detector_from_webcam():
    """
    Continuously captures frames from the webcam and runs YOLOv8 detection.
    The loop is controlled by a manual stop button.
    """
    print("Webcam is ready. Click the 'STOP Detection' button to end the feed.")

    stop_detection = False

    # status updates ("Capturing...", "Processing...")
    status_label = HTML(value="<p style='color: blue;'>Waiting for capture to start...</p>")

  #stop button widget
    def on_button_click(b):
        nonlocal stop_detection
        stop_detection = True
        status_label.value = "<p style='color: red;'>Stopping detection loop...</p>"
        b.description = "Detection Stopped"
        b.disabled = True

    stop_button = Button(
        description='STOP Detection',
        disabled=False,
        button_style='danger',
        tooltip='Click to stop the live feed processing.',
        layout=Layout(width='200px', height='40px')
    )
    stop_button.on_click(on_button_click)

    # ipywidgets.Image to display the video frame
    frame_image = Image(
        format='jpeg',
        layout=Layout(width='800px', height='450px', border='2px solid #3b82f6')
    )

    # VBox to display all elements once, preventing the blinking
    app_container = VBox([stop_button, status_label, frame_image])
    display.display(app_container)


    try:
        # Loop to capture, process, and display frames until the stop button is pressed
        while not stop_detection:
            # 1. Capture Frame (This will open the browser's camera permission prompt)
            status_label.value = "<p style='color: orange;'>Capturing frame (please wait for camera prompt)...</p>"
            frame = get_webcam_frame()
            if frame is None:
                status_label.value = "<p style='color: red;'>Error: Could not capture frame.</p>"
                break

            # 2. Perform object detection
            status_label.value = "<p style='color: green;'>Processing frame with YOLOv8...</p>"
            results = model.predict(frame, conf=0.5, iou=0.7, verbose=False)

            # 3. Get the frame with the detection boxes plotted on it.
            annotated_frame = results[0].plot()

            # 4. Convert the OpenCV frame (numpy array) to JPEG bytes for the Image widget
            resized_frame = cv2.resize(annotated_frame, (800, 450))
            is_success, buffer = cv2.imencode(".jpg", resized_frame)

            if is_success:
                # 5. Update the Image widget in place
                frame_image.value = buffer.tobytes()

            status_label.value = "<p style='color: blue;'>Displaying frame. Waiting for next capture...</p>"


            time.sleep(0.5)

        print("Application closed.")

    except Exception as e:
        status_label.value = f"<p style='color: red;'>An unexpected error occurred: {e}</p>"
        print(f"An unexpected error occurred during detection: {e}")

    finally:
        print("Releasing resources...")
        print("Cleanup complete.")
#run
if __name__ == "__main__":
    run_detector_from_webcam()

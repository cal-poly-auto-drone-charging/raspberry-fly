from picamera2 import Picamera2
import numpy as np
import cv2
import threading

class PiCameraStream:
    def __init__(self, resolution=(1920, 1080), framerate=30):
        self.picam = Picamera2()
        self.resolution = resolution
        self.framerate = framerate
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

        # Configure the camera
        config = self.picam.create_video_configuration(main={"size": self.resolution, "format": "XRGB8888"})
        self.picam.configure(config)

    def start_stream(self):
        self.stopped = False
        self.picam.start_preview()  # This might be necessary for initializing the camera pipeline
        self.picam.start()
        threading.Thread(target=self.update, args=(), daemon=True).start()

    def update(self):
        while not self.stopped:
            try:
                # Fetch the next completed request from the camera system.
                # This method returns a CompletedRequest object directly.
                request = self.picam.capture_request()

                if request:  # Check if the request is valid
                    with self.lock:
                        # Convert the request buffer to an image array and store it.
                        self.frame = cv2.cvtColor(request.make_array("main"), cv2.COLOR_RGB2BGR)

                    # After processing this request, release it to free up resources.
                    request.release()
            except Exception as e:
                print(f"Error capturing frame: {e}")

    def read_frame(self):
        with self.lock:
            return self.frame

    def stop_stream(self):
        self.stopped = True
        self.picam.stop_preview()
        self.picam.stop()

    def release(self):
        self.stop_stream()
        self.picam.close()

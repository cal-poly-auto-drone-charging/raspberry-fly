import unittest
import time
import numpy as np
from pi_camera_stream import PiCameraStream  # Replace with the actual module name

class TestPiCameraStream(unittest.TestCase):

    def setUp(self):
        self.camera_stream = PiCameraStream()
        self.camera_stream.start_stream()
        time.sleep(2)  # Allow some time for the camera to initialize

    def tearDown(self):
        self.camera_stream.release()

    def test_frame_retrieval_speed(self):
        num_frames = 10
        start_time = time.time()

        for _ in range(num_frames):
            frame = self.camera_stream.read_frame()
            self.assertIsNotNone(frame)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000000  # Convert to milliseconds
        avg_time_per_frame = total_time_ms / num_frames

        print(f"Average time per frame: {avg_time_per_frame:.2f} us")

if __name__ == '__main__':
    unittest.main()

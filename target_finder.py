import cv2
import numpy as np

class TargetFinder:
    def __init__(self, tracker, spotter, frame_width, frame_height, annotate=False):
        self.tracker = tracker
        self.spotter = spotter
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.annotate = annotate

        # Control variables
        self.use_tracker = False
        self.spotter_success_cycles = 0
        self.spotter_failure_cycles = 0
        self.tracker_failures = 0
        self.blur_sigma = None

        # Thresholds
        self.tracker_fail_thresh = 5
        self.tracker_wait_period = 20
        self.blur_success_thresh = 2
        self.spotter_confidence_thresh = 5
        self.spotter_duration_thresh = 4
        self.spotter_area_thresh = 110

    def process_frame(self, frame):
        if self.spotter_success_cycles == 0:
            self.blur_sigma = None
        corners = None
        if self.use_tracker:
            H = self.tracker.compute_homography(frame)
            if H is not None:
                corners = self.tracker.get_corners(H)
                self.tracker_failures = 0
            else:
                self.tracker_failures += 1
                if self.tracker_failures >= self.tracker_fail_thresh:
                    self.use_tracker = False
        else:
            best_rect_info = self.spotter.get_best_rect(frame, self.blur_sigma)
            if best_rect_info:
                corners = self.handle_spotter_success(best_rect_info)
            else:
                self.handle_spotter_failure()

        if self.annotate:
            frame = self.annotate_frame(frame, corners)

        return corners, frame

    def handle_spotter_success(self, best_rect_info):
        best_rect, score = best_rect_info
        box = cv2.boxPoints(best_rect)
        area = cv2.contourArea(box)

        if self.spotter_success_cycles > self.blur_success_thresh:
            self.blur_sigma = np.sqrt(area)
        if score > self.spotter_confidence_thresh and area > self.frame_width * self.frame_height / self.spotter_area_thresh:
            self.spotter_success_cycles += 1
            if self.spotter_success_cycles >= self.spotter_duration_thresh:
                self.use_tracker = True
                self.tracker_failures = 0

        return box

    def handle_spotter_failure(self):
        self.spotter_success_cycles = 0
        self.spotter_failure_cycles += 1
        if self.spotter_failure_cycles >= self.tracker_wait_period:
            self.tracker_failures = self.tracker_fail_thresh - 1
            self.spotter_failure_cycles = 0
            self.use_tracker = True

    def annotate_frame(self, frame, corners):
        if corners is not None:
            corners = np.intp(corners)
            cv2.drawContours(frame, [corners], 0, (0, 255, 0), 2)
        return frame

    def calculate_spherical_coordinates(self, corners, focal_length, pitch, frame_size, target_size):
        """
        Calculate the spherical coordinates (radius, azimuth, elevation) of the target.

        Parameters:
        corners: Corners of the target in the image.
        focal_length: Focal length of the camera.
        pitch: Pitch angle of the camera (in degrees).
        frame_size: Size of the frame (width, height).
        target_size: Known size of the target (width, height).

        Returns:
        (radius, azimuth, elevation): Spherical coordinates of the target center.
        """
        # Convert pitch to radians
        pitch = np.radians(pitch)

        # Calculate the center of the target in image coordinates
        center_x, center_y = np.mean(corners, axis=0)

        # Calculate the physical size of the target in the image
        pixel_width = np.linalg.norm(corners[0] - corners[1])
        pixel_height = np.linalg.norm(corners[0] - corners[3])

        # Calculate the distance to the target (radius)
        actual_width, actual_height = target_size
        width_ratio = actual_width / pixel_width
        height_ratio = actual_height / pixel_height
        distance_width = width_ratio * focal_length
        distance_height = height_ratio * focal_length
        radius = (distance_width + distance_height) / 2  # Average distance

        # Calculate azimuth and elevation
        frame_width, frame_height = frame_size
        x = (center_x - frame_width / 2) * width_ratio
        y = (center_y - frame_height / 2) * height_ratio
        azimuth = np.arctan2(x, radius)
        elevation = np.arctan2(y, radius) - pitch

        return radius, np.degrees(azimuth), np.degrees(elevation)
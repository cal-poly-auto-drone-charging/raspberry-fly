import cv2
import numpy as np

class TargetFinder:
    def __init__(self, tracker, spotter, frame_width, frame_height, annotate=False, 
                 focal_length=0.00304, target_dimensions=(0.2032,0.2032), camera_pitch=-90):
        self.tracker = tracker
        self.spotter = spotter
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.annotate = annotate

        # Calibrations
        self.focal_length = focal_length
        self.target_dimensions = target_dimensions
        self.camera_pitch = camera_pitch

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

        # Reshape the box to have the same 3D shape as the tracker's output
        box_reshaped = box.reshape(-1, 1, 2)
        return box_reshaped


    def handle_spotter_failure(self):
        self.spotter_success_cycles = 0
        self.spotter_failure_cycles += 1
        if self.spotter_failure_cycles >= self.tracker_wait_period:
            self.tracker_failures = self.tracker_fail_thresh - 1
            self.spotter_failure_cycles = 0
            self.use_tracker = True

    def annotate_frame(self, frame, corners):
        # Calculate cylindrical coordinates if corners are available
        if corners is not None:
            radius, azimuth, height = self.calculate_cylindrical_coordinates(corners)
            if radius is not None:
                corners = np.intp(corners)
                cv2.drawContours(frame, [corners], 0, (0, 255, 0), 2)
                # Format the float values to one decimal place
                radius = f"{radius:.1f}"
                azimuth = f"{azimuth:.1f}"
                height = f"{height:.1f}"
            else:
                radius = azimuth = height = 'N/A'
        else:
            radius = azimuth = height = 'N/A'

        t_string = "Tracker"
        s_string = "Spotter"

        # Text to be displayed
        text_lines = [
            f"Strategy: {t_string if self.use_tracker else s_string}",
            f"Handoff Readiness: {self.spotter_success_cycles}",
            f"Spotter Miss Count: {self.spotter_failure_cycles}",
            f"Tracker Miss Count: {self.tracker_failures}",
            f"Blur Sigma: {self.blur_sigma:.1f}" if self.blur_sigma is not None else "Blur Sigma: N/A",
            "",
            f"Radius : {radius}",
            f"Azimuth: {azimuth}",
            f"Height : {height}"
        ]

        # Display each line of text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 1
        line_type = cv2.LINE_AA
        x, y = 10, 30  # Starting position

        for line in text_lines:
            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, line_type)
            y += 20  # Move to the next line

        return frame

    def calculate_cylindrical_coordinates(self, corners):
        if corners is None or len(corners) != 4:
            return None, None, None

        # Reshape corners to 2D if necessary (assuming shape is [4, 1, 2])
        if corners.ndim == 3:
            corners_2d = corners.reshape(-1, 2)
        else:
            corners_2d = corners

        width, height = self.target_dimensions
        pixel_width = np.linalg.norm(corners_2d[0] - corners_2d[1])
        pixel_height = np.linalg.norm(corners_2d[0] = corners_2d[2])
        scale_factor = (width / pixel_width) if (pixel_width >= pixel_height) else (height / pixel_height)
        # Calculate distance to target
        distance = self.focal_length * scale_factor


        # Calculate the center of the rectangle
        center = np.mean(corners_2d, axis=0)
        center_x, center_y = center[0], center[1]

        # Calculate azimuth
        frame_center_x, frame_center_y = self.frame_width / 2, self.frame_height / 2
        azimuth = np.arctan2(center_y - frame_center_y, center_x - frame_center_x)

        # Calculate radius
        radius = 0

        # Estimate height using target dimensions and rectangle size in the image
        estimated_height = scale_factor * self.frame_width

        return radius, np.degrees(azimuth) + 90, estimated_height


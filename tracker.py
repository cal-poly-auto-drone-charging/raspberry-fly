import cv2
import numpy as np

class Tracker:
    DEFAULT_MIN_MATCH_COUNT = 10
    DEFAULT_INLIER_THRESHOLD = 5.0
    DEFAULT_RATIO_THRESH = 0.75

    def __init__(self, reference, min_match_count=DEFAULT_MIN_MATCH_COUNT, 
                 inlier_threshold=DEFAULT_INLIER_THRESHOLD):
        """ Initialize the Tracker with a reference image and parameters for matching. """
        self.min_match_count = min_match_count
        self.inlier_threshold = inlier_threshold
        self.reference = reference

        self.sift = cv2.SIFT_create()  # Single SIFT instance for reuse
        self.bf = cv2.BFMatcher()      # Single BFMatcher instance for reuse

        # Compute keypoints and descriptors for the reference image
        self.reference_keypoints, self.reference_descriptors = self.compute_keypoints_and_descriptors(reference)

    def compute_keypoints_and_descriptors(self, image):
        """ Compute SIFT keypoints and descriptors for a given image. """
        return self.sift.detectAndCompute(image, None)

    def compute_homography(self, frame, ratio_thresh=DEFAULT_RATIO_THRESH):
        """ Compute homography between the reference image and a frame. """
        frame_keypoints, frame_descriptors = self.compute_keypoints_and_descriptors(frame)
        if frame_descriptors is None:
            return None
        
        good_matches = self.find_good_matches(frame_descriptors, ratio_thresh)

        if len(good_matches) > self.min_match_count and good_matches is not None:
            return self.calculate_homography(good_matches, frame_keypoints)
        return None

    def find_good_matches(self, descriptors, ratio_thresh):
        """ Find good matches using the ratio test. """
        matches = self.bf.knnMatch(self.reference_descriptors, descriptors, k=2)
        if not matches:
            return None
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        return good_matches

    def calculate_homography(self, good_matches, frame_keypoints):
        """ Calculate the homography using matched keypoints. """
        src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.inlier_threshold)[0]

    def get_corners(self, H):
        """ Calculate the transformed corners of the reference image using homography. """
        ref_h, ref_w = self.reference.shape[:2]
        ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(ref_corners, H) if H is not None else None

    def augment_frame(self, frame, H, yaw='--', altitude='--', distance='--'):
        """ Draw border and add text on video frame based on homography.
            
            Arguments:
                frame: frame to be drawn on [H,W,3]
                H: homography [3,3]
                yaw: Yaw value as a string
                altitude: Altitude value as a string
                distance: Distance value as a string
            Returns:
                frame with border and text [H,W,3]
        """
        # Define the corners of the reference image
        ref_h, ref_w = self.reference.shape[:2]
        ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)

        target_spotted = "Target Spotted: Yes" if H is not None else "Target Spotted: No"
        corner_texts = ["Top Left: --", "Top Right: --", "Bottom Right: --", "Bottom Left: --"]

        if H is not None:
            # Transform the corners with the homography
            transformed_corners = cv2.perspectiveTransform(ref_corners, H)

            # Draw the border
            num_corners = len(transformed_corners)
            for i in range(num_corners):
                pt1 = tuple(transformed_corners[i][0].astype(int))
                pt2 = tuple(transformed_corners[(i+1) % num_corners][0].astype(int))
                color = (0, 0, 255) if i == 0 else (255, 0, 0)  # Red for top, blue for others
                cv2.line(frame, pt1, pt2, color, thickness=5)

            # Update corner text with actual coordinates
            corner_texts = [
                f"Top Left: {tuple(transformed_corners[0][0].astype(int))}",
                f"Top Right: {tuple(transformed_corners[1][0].astype(int))}",
                f"Bottom Right: {tuple(transformed_corners[2][0].astype(int))}",
                f"Bottom Left: {tuple(transformed_corners[3][0].astype(int))}"
            ]

        # Text settings
        text_color = (0, 255, 0)  # Green color for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2

        # Put the text on the frame
        y_offset = 30  # Initial y-coordinate for text
        line_height = 30  # Height of each line of text

        cv2.putText(frame, target_spotted, (10, y_offset), font, font_scale, text_color, thickness, cv2.LINE_AA)
        for i, text in enumerate(corner_texts):
            cv2.putText(frame, text, (10, y_offset + (i + 2) * line_height), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Additional info (Yaw, Altitude, Distance)
        additional_texts = [f"Yaw: {yaw}", f"Altitude: {altitude}", f"Distance: {distance}"]
        for i, text in enumerate(additional_texts):
            cv2.putText(frame, text, (10, y_offset + (i + 7) * line_height), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return frame.astype(np.uint8)



import cv2 
import numpy as np

class Tracker:
    def __init__(self,reference,min_match_count=10,inlier_threshold=5.0):
        """ Initializes a Tracker object.
            
            During initialization, this function will compute and store SIFT keypoints
            for the reference image.

            Arguments:
                reference: reference image
                min_match_count: minimum number of matches for a video frame to be processed.
                inlier_threshold: maximum re-projection error for inliers in homography computation
        """
        self.min_match_count = min_match_count
        self.inlier_threshold = inlier_threshold
        self.reference = reference

        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        self.reference_keypoints, self.reference_descriptors = sift.detectAndCompute(reference, None)
        
    def compute_homography(self, frame, ratio_thresh=0.75):
        """ Calculate homography relating the reference image to a query frame using OpenCV's RANSAC.
        
            Arguments:
                frame: query frame from video
                ratio_thresh: ratio threshold for filtering keypoint matches
            Returns:
                the estimated homography [3,3] or None if not enough matches are found
        """
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors in the frame
        frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)

        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.reference_descriptors, frame_descriptors, k=2)

        # Apply ratio test
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

        if len(good_matches) > self.min_match_count:
            src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography using OpenCV's RANSAC method
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.inlier_threshold)

            return H
        else:
            return None

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



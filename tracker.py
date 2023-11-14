import cv2 
import numpy as np

class Tracker:
    def __init__(self,reference,overlay,min_match_count=10,inlier_threshold=5):
        """ Initializes a Tracker object.
            
            During initialization, this function will compute and store SIFT keypoints
            for the reference image.

            Arguments:
                reference: reference image
                overlay: overlay image for augmented reality effect
                min_match_count: minimum number of matches for a video frame to be processed.
                inlier_threshold: maximum re-projection error for inliers in homography computation
        """
        self.min_match_count = min_match_count
        self.inlier_threshold = inlier_threshold
        self.reference = reference

        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        self.reference_keypoints, self.reference_descriptors = sift.detectAndCompute(reference, None)
        
    def compute_homography(self, frame, ratio_thresh=0.7):
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
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            return H
        else:
            return None

    def augment_frame(self, frame, H):
        """ Draw border on video frame based on homography, with the top border in red.
            
            Arguments:
                frame: frame to be drawn on [H,W,3]
                H: homography [3,3]
            Returns:
                frame with border [H,W,3]
        """
        # Get dimensions of the frame
        h, w = frame.shape[:2]

        # Define the corners of the reference image
        ref_h, ref_w = self.reference.shape[:2]
        ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)

        # Transform the corners with the homography
        transformed_corners = cv2.perspectiveTransform(ref_corners, H)

        # Draw a red line for the top border
        pt1 = tuple(transformed_corners[0][0].astype(int))
        pt2 = tuple(transformed_corners[1][0].astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 255), thickness=5)  # Red line for the top border

        # Draw blue lines for the other sides
        for i in range(1, len(transformed_corners)):
            pt1 = tuple(transformed_corners[i][0].astype(int))
            pt2 = tuple(transformed_corners[(i+1) % len(transformed_corners)][0].astype(int))
            cv2.line(frame, pt1, pt2, (255, 0, 0), thickness=5)  # Blue lines for the other sides

        return frame.astype(np.uint8)


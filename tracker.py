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
        self.overlay = overlay

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
    
    def augment_frame(self,frame,H):
        """ Draw overlay image on video frame.
            
            Arguments:
                frame: frame to be drawn on [H,W,3]
                H: homography [3,3]
            Returns:
                augmented frame [H,W,3]
        """
        # Get dimensions of the frame and the overlay
        h, w = frame.shape[:2]
        overlay_h, overlay_w = self.overlay.shape[:2]

        # Create an all-white image of the same size as the overlay
        white_overlay = np.ones_like(self.overlay) * 255

        # Warp the white overlay image using the homography to create an alpha mask
        alpha_mask = cv2.warpPerspective(white_overlay, H, (w, h))

        # Warp the actual overlay image
        warped_overlay = cv2.warpPerspective(self.overlay, H, (w, h))

        # Use the alpha mask to blend the warped overlay and the original frame
        # Normalize alpha mask to range [0, 1]
        alpha_mask = alpha_mask.astype(float) / 255
        # Blend the original frame and the warped overlay using the alpha mask
        augmented_frame = (1 - alpha_mask) * frame + alpha_mask * warped_overlay

        # Define the corners of the overlay image
        overlay_corners = np.float32([[0, 0], [overlay_w, 0], [overlay_w, overlay_h], [0, overlay_h]]).reshape(-1, 1, 2)

        # Transform the corners with the homography
        transformed_corners = cv2.perspectiveTransform(overlay_corners, H)

        # Draw a thick blue border around the transformed corners
        num_corners = len(transformed_corners)
        for i in range(num_corners):
            pt1 = tuple(transformed_corners[i][0].astype(int))
            pt2 = tuple(transformed_corners[(i+1) % num_corners][0].astype(int))
            cv2.line(augmented_frame, pt1, pt2, (255, 0, 0), thickness=5)

        return augmented_frame.astype(np.uint8)


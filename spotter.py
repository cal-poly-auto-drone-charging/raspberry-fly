import cv2
import numpy as np

class Spotter:
    def __init__(self, resize_width, resize_height, white_thresholds=((0, 0, 220), (172, 50, 255)), blur_kernel_size=(5, 5), edge_thresholds=(50, 150), min_area=100, max_area=5000, aspect_ratio_range=(0.7, 1.3), morph_kernel_size=(5, 5)):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.white_thresholds = white_thresholds
        self.blur_kernel_size = blur_kernel_size
        self.edge_thresholds = edge_thresholds
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.morph_kernel_size = morph_kernel_size

    def process_frame(self, frame):
        resized_frame = self.resize_frame(frame)
        contrast_frame = self.enhance_contrast(resized_frame)
        white_filtered_frame = self.filter_white_color(resized_frame)
        edges = self.detect_edges(white_filtered_frame)
        contours = self.find_contours(edges)

        return contours

    def resize_frame(self, frame):
        return cv2.resize(frame, (self.resize_width, self.resize_height))

    def enhance_contrast(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized_gray = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)

    def filter_white_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white, upper_white = self.white_thresholds
        mask = cv2.inRange(hsv, np.array(lower_white, dtype=np.uint8), np.array(upper_white, dtype=np.uint8))

        # Apply morphological opening and closing to remove noise
        kernel = np.ones(self.morph_kernel_size, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return cv2.bitwise_and(frame, frame, mask=mask)

    def detect_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        return cv2.Canny(blurred, self.edge_thresholds[0], self.edge_thresholds[1])


    def find_contours(self, edge_frame):
        contours, _ = cv2.findContours(edge_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

                if len(approx) == 4:  # Check if the approximated contour has 4 sides
                    (x, y, w, h) = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                        filtered_contours.append(contour)

        return filtered_contours

    def debug_frame(self, frame):
        resized_frame = self.resize_frame(frame)
        contrast_frame = self.enhance_contrast(resized_frame)
        white_filtered_frame = self.filter_white_color(contrast_frame)
        edges = self.detect_edges(white_filtered_frame)
        contours = self.find_contours(edges)

        # Creating copies for drawing
        contour_frame = contrast_frame.copy()
        rectangle_frame = contrast_frame.copy()

        # Draw contours
        cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)

        # Draw rectangles around detected objects
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(rectangle_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Combine frames into a single image for debugging
        top_row = np.hstack((resized_frame, white_filtered_frame))
        bottom_row = np.hstack((contour_frame, rectangle_frame))
        debug_image = np.vstack((top_row, bottom_row))

        return debug_image

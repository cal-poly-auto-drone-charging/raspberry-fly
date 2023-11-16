import cv2
import numpy as np

class Spotter:
    def __init__(self, resize_width, resize_height, white_thresholds=((0, 0, 220), (172, 50, 255)), blur_kernel_size=(5, 5), edge_thresholds=(50, 150), min_area=200, max_area=10000, aspect_ratio_range=(0.7, 1.3), morph_kernel_size=(5, 5), rect_area_ratio=0.9):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.white_thresholds = white_thresholds
        self.blur_kernel_size = blur_kernel_size
        self.edge_thresholds = edge_thresholds
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.morph_kernel_size = morph_kernel_size
        self.rect_area_ratio = rect_area_ratio
        self.rect_history = []

    def process_frame(self, frame):
        resized_frame = self.resize_frame(frame)
        contrast_frame = self.enhance_contrast(resized_frame)
        white_filtered_frame = self.filter_white_color(resized_frame)
        gray = cv2.cvtColor(white_filtered_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        edge_frame = cv2.Canny(blurred, self.edge_thresholds[0], self.edge_thresholds[1])
        contours, _ = cv2.findContours(edge_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            rect_area = rect_width * rect_height
            if self.min_area <= rect_area <= self.max_area:
                aspect_ratio = rect_width / rect_height if rect_width < rect_height else rect_height / rect_width
                if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                    rectangles.append(rect)

        # Score and sort rectangles
        scored_rectangles = [(rect, self.calculate_score(rect)) for rect in rectangles]
        scored_rectangles.sort(key=lambda x: x[1], reverse=True)

        # Update history
        self.rect_history.insert(0, [rect for rect, _ in scored_rectangles[:5]])
        if len(self.rect_history) > 10:
            self.rect_history.pop()

        # Return sorted rectangles
        return [rect for rect, _ in scored_rectangles]

    def get_best_rect(self, frame):
        scored_rectangles = self.process_frame(frame)
        print(scored_rectangles)
        if scored_rectangles:
            return scored_rectangles[0]
        else:
            return None

    def calculate_score(self, rect, cycle_decay=0.9, top_rectangles=5):
        score = 0
        for cycle, history in enumerate(self.rect_history):
            decay_factor = cycle_decay ** cycle
            for past_rect in history[:top_rectangles]:
                score += decay_factor * self.similarity_score(rect, past_rect)
        return score

    def similarity_score(self, rect1, rect2, position_weight=1.0, size_weight=1.0, angle_weight=1.0):
        center1, size1, angle1 = rect1
        center2, size2, angle2 = rect2

        # Position similarity
        distance = np.linalg.norm(np.array(center1) - np.array(center2))

        # Size similarity
        area1 = size1[0] * size1[1]
        area2 = size2[0] * size2[1]
        size_difference = abs(area1 - area2)

        # Angle similarity
        angle_difference = abs(angle1 - angle2)

        # Normalized inverse scores
        position_score = 1 / (1 + distance)
        size_score = 1 / (1 + size_difference)
        angle_score = 1 / (1 + angle_difference)

        # Weighted sum of scores
        total_score = (position_score * position_weight +
                       size_score * size_weight +
                       angle_score * angle_weight)

        return total_score

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
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
            area = cv2.contourArea(approx)
            if area <= 25 or area > 10000 or len(approx) < 4:
                continue
            rect = cv2.minAreaRect(approx)
            rect_width, rect_height = rect[1]
            rect_area = rect_width * rect_height

            if rect_area > 0:  # To avoid division by zero
                if area >= (rect_area * self.rect_area_ratio):
                    aspect_ratio = rect_width / rect_height
                    if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                        filtered_contours.append(contour)
        return filtered_contours


    def debug_frame(self, frame):
        resized_frame = self.resize_frame(frame)
        contrast_frame = self.enhance_contrast(resized_frame)
        white_filtered_frame = self.filter_white_color(contrast_frame)

        rsz_edges = self.detect_edges(resized_frame)
        rsz_contours = self.find_contours(rsz_edges)

        cont_edges = self.detect_edges(contrast_frame)
        cont_contours = self.find_contours(cont_edges)

        whfl_edges = self.detect_edges(white_filtered_frame)
        whfl_contours = self.find_contours(whfl_edges)

        # Creating copies for drawing
        rsz_frame = contrast_frame.copy()
        cont_frame = contrast_frame.copy()
        whfl_frame = contrast_frame.copy()

        # Draw contours
        cv2.drawContours(rsz_frame, rsz_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(cont_frame, cont_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(whfl_frame, whfl_contours, -1, (0, 255, 0), 2)

        # Combine frames into a single image for debugging
        top_row = np.hstack((resized_frame, rsz_frame))
        bottom_row = np.hstack((cont_frame, whfl_frame))
        debug_image = np.vstack((top_row, bottom_row))

        return debug_image

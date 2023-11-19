import cv2
import numpy as np

class Spotter:
    HISTORY_SIZE = 20
    SCORE_CYCLE_DECAY = 0.9
    SCORE_TOP_RECTANGLES = 5
    SIMILARITY_POSITION_WEIGHT = 2.0
    SIMILARITY_SIZE_WEIGHT = 1.0
    SIMILARITY_ANGLE_WEIGHT = 1.0

    def __init__(self, resize_width, resize_height, 
                 white_thresholds=((0, 0, 220), (172, 50, 255)), 
                 blur_kernel_size=(75, 75), blur_sigma=1, 
                 edge_thresholds=(50, 150), min_area=200, 
                 max_area=100000, aspect_ratio_range=(0.7, 1.3), 
                 morph_kernel_size=(5, 5), rect_area_ratio=0.9):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.white_thresholds = white_thresholds
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.edge_thresholds = edge_thresholds
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.morph_kernel_size = morph_kernel_size
        self.rect_area_ratio = rect_area_ratio
        self.rect_history = []

    def process_frame(self, frame, blur_sigma=None):
        """ Process a frame to detect and score rectangles. """
        if blur_sigma is None:
            blur_sigma = self.blur_sigma
        resized_frame = self.resize_frame(frame)
        white_filtered_frame = self.filter_white_color(resized_frame)
        gray = cv2.cvtColor(white_filtered_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, blur_sigma)
        edge_frame = cv2.Canny(blurred, *self.edge_thresholds)
        contours, _ = cv2.findContours(edge_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = self.extract_valid_rectangles(contours)
        scored_rectangles = [(rect, self.calculate_score(rect)) for rect in rectangles]
        scored_rectangles.sort(key=lambda x: x[1], reverse=True)
        self.update_history(scored_rectangles)

        return scored_rectangles

    def get_best_rect(self, frame, blur_sigma=None):
        """ Get the best rectangle for a given frame. """
        if blur_sigma is None:
            blur_sigma = self.blur_sigma
        scored_rectangles = self.process_frame(frame, blur_sigma)
        return scored_rectangles[0] if scored_rectangles else None

    def calculate_score(self, rect, cycle_decay=SCORE_CYCLE_DECAY, top_rectangles=SCORE_TOP_RECTANGLES):
        """ Calculate the score for a rectangle based on its history. """
        score = 0
        for cycle, history in enumerate(self.rect_history):
            decay_factor = cycle_decay ** cycle
            for past_rect in history[:top_rectangles]:
                score += decay_factor * self.similarity_score(rect, past_rect)
        return score

    def similarity_score(self, rect1, rect2, 
                         position_weight=SIMILARITY_POSITION_WEIGHT, 
                         size_weight=SIMILARITY_SIZE_WEIGHT, 
                         angle_weight=SIMILARITY_ANGLE_WEIGHT):
        """ Calculate the similarity score between two rectangles. """
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

        position_score = 1 / (1 + distance)
        size_score = 1 / (1 + size_difference)
        angle_score = 1 / (1 + angle_difference)

        total_score = (position_score * position_weight +
                       size_score * size_weight +
                       angle_score * angle_weight)
        return total_score

    def resize_frame(self, frame):
        """ Resize the frame to the specified dimensions. """
        return cv2.resize(frame, (self.resize_width, self.resize_height))

    def filter_white_color(self, frame):
        """ Filter white color in the frame. """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white, upper_white = self.white_thresholds
        mask = cv2.inRange(hsv, np.array(lower_white, dtype=np.uint8),
                           np.array(upper_white, dtype=np.uint8))
        kernel = np.ones(self.morph_kernel_size, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def extract_valid_rectangles(self, contours):
        """ Extract valid rectangles from contours. """
        rectangles = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            if self.is_valid_rectangle(rect):
                rectangles.append(rect)
        return rectangles

    def is_valid_rectangle(self, rect):
        """ Check if a rectangle is valid based on size and aspect ratio. """
        rect_width, rect_height = rect[1]
        rect_area = rect_width * rect_height
        if not (self.min_area <= rect_area <= self.max_area):
            return False

        aspect_ratio = min(rect_width, rect_height) / max(rect_width, rect_height)
        return self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]

    def update_history(self, scored_rectangles):
        """ Update the history with the latest rectangles. """
        self.rect_history.insert(0, [rect for rect, _ in scored_rectangles[:self.SCORE_TOP_RECTANGLES]])
        if len(self.rect_history) > self.HISTORY_SIZE:
            self.rect_history.pop()

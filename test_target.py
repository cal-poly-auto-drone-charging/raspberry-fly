import cv2
import numpy as np
from tracker import Tracker
from spotter import Spotter  # Import Spotter class
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='S1000644.LRV')
parser.add_argument('--reference', default='qr_code.png')
parser.add_argument('--output', default='output.mp4')
args = parser.parse_args()

ref1 = cv2.imread(args.reference)

cap = cv2.VideoCapture(args.input)

# Get video properties for the output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

tracker = Tracker(ref1)
spotter = Spotter(frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, frame_fps, (frame_width, frame_height))

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    if frame_number < frame_fps * 45:
        continue

    # Compute homography and augment frame
    H = tracker.compute_homography(frame)
    frame_out = tracker.augment_frame(frame, H)

    # Get the best rectangle for the current frame
    best_rect_info = spotter.get_best_rect(frame)

    # Draw the best rectangle on the frame, if it exists
    if best_rect_info is not None:
        best_rect, score = best_rect_info
        box = cv2.boxPoints(best_rect)
        box = np.intp(box)
        cv2.drawContours(frame_out, [box], 0, (0, 255, 0), 2)

    out.write(frame_out)  # Write frame to output video
    cv2.imshow('window', frame_out)

    if cv2.waitKey(1) == 27:  # hit escape to exit
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

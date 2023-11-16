import cv2
import numpy as np
from tracker import Tracker
from spotter import Spotter  # Import Spotter class
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='S1000634.LRV')
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
spotter = Spotter(frame_width, frame_height)  # Initialize Spotter with desired width and height
print(frame_fps)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, frame_fps, (frame_width, frame_height))

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # if frame_number < 30 * 51:
    #     continue
    frame_number += 1

    H = tracker.compute_homography(frame)
    frame_out = tracker.augment_frame(frame, H)
    contours = spotter.process_frame(frame)  # Use Spotter to process the frame
    # frame_out = frame.copy()
    cv2.drawContours(frame_out, contours, -1, (0, 255, 0), 2)  # Draw contours on the frame

    out.write(frame_out)  # Write frame to output video
    cv2.imshow('window', frame_out)

    if cv2.waitKey(1) == 27:  # hit escape to exit
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

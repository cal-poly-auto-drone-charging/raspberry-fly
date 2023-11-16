import cv2
import numpy as np
from tracker import Tracker
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='IMG_1476.MOV')
parser.add_argument('--reference', default='qr_code_bk.png')
parser.add_argument('--output', default='output.mp4')  # Add output argument
args = parser.parse_args()

ref1 = cv2.imread(args.reference)

tracker = Tracker(ref1)

cap = cv2.VideoCapture(args.input)

# Get video properties for the output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(args.output, fourcc, frame_fps, (frame_width, frame_height))

frame_counter = 0  # Initialize frame counter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1  # Increment frame counter

    # Process only 1 out of every 20 frames
    if True:
        H = tracker.compute_homography(frame)
        if H is None:
            # Output from spotter here
        else:
            frame_out = tracker.augment_frame(frame, H)

        out.write(frame_out)  # Write frame to output video
        cv2.imshow('window', frame_out)

    if cv2.waitKey(1) == 27:  # hit escape to exit
        break

# Release everything if job is finished
cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()

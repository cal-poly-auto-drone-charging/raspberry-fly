import cv2
import numpy as np
from tracker import Tracker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='S1000644.LRV')
parser.add_argument('--reference', default='qr_code.png')
parser.add_argument('--overlay', default='overlay.png')
args = parser.parse_args()

ref = cv2.imread(args.reference)
overlay = cv2.imread(args.overlay)

tracker = Tracker(ref, overlay)

cap = cv2.VideoCapture(args.input)

frame_counter = 0  # Initialize frame counter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1  # Increment frame counter

    # Process only 1 out of every 20 frames
    if frame_counter > 180*11:
        H = tracker.compute_homography(frame)
        if H is not None:
            frame_out = tracker.augment_frame(frame, H)
        else:
            frame_out = frame

        cv2.imshow('window', frame_out)

    if cv2.waitKey(1) == 27:  # hit escape to exit
        break

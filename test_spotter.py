import cv2
import numpy as np
from spotter import Spotter
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='S1000644.LRV')
parser.add_argument('--output', default='output.mp4')
args = parser.parse_args()

# Open the video
cap = cv2.VideoCapture(args.input)

# Get video properties for the output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

print(frame_fps)

# Initialize Spotter with desired parameters (modify these as needed)
spotter = Spotter(resize_width=frame_width, resize_height=frame_height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(args.output, fourcc, frame_fps, (frame_width, frame_height))

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    if frame_number > 60*51
    # Process the frame with the Spotter
    processed_frame = spotter.debug_frame(frame)

    # Write the processed frame to the output video
    out.write(processed_frame)

    # Optionally display the frame
    cv2.imshow('Processed Frame', processed_frame)

    if cv2.waitKey(1) == 27:  # hit escape to quit
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

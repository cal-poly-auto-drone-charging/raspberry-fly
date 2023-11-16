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

    if frame_number < frame_fps * 40:
        continue

    # Get the best rectangle for the current frame
    best_rect_info = spotter.get_best_rect(frame)
    
    # Draw the best rectangle on the frame, if it exists
    if best_rect_info is not None:
        best_rect, score = best_rect_info
        print(score)
        box = cv2.boxPoints(best_rect)
        box = np.intp(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    # Write the frame with the drawn rectangle to the output video
    out.write(frame)

    # Optionally display the frame
    cv2.imshow('Frame with Best Rectangle', frame)

    if cv2.waitKey(1) == 27:  # hit escape to quit
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

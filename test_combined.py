import cv2
import numpy as np
from tracker import Tracker
from spotter import Spotter  # Import Spotter class
import argparse
import time

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
use_tracker = False  # Initially, don't use tracker
spotter_success_cycles = 0  # Count successful cycles for spotter
tracker_failures = 0  # Count failures for tracker
blur_sigma = None

# thresholds for test
tracker_fail_thresh = 5
blur_success_thresh = 2
spotter_confidence_thresh = 5
spotter_duration_thresh = 4
spotter_area_thresh = 110

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    if frame_number < frame_fps * 40:
        continue

    if spotter_success_cycles == 0:
        blur_sigma = None
        pass

    start_time = time.time()

    # Spotter and Tracker logic
    if use_tracker:
        H = tracker.compute_homography(frame)
        if H is not None:
            frame_out = tracker.augment_frame(frame, H)
            tracker_failures = 0  # Reset failure count
        else:
            tracker_failures += 1
            frame_out = frame
            if tracker_failures >= tracker_fail_thresh:
                use_tracker = False  # Switch back to spotter
                #spotter_success_cycles -= 3
    else:
        best_rect_info = spotter.get_best_rect(frame, blur_sigma)
        frame_out = frame  # Default output is the original frame

        if best_rect_info is not None:
            best_rect, score = best_rect_info
            box = cv2.boxPoints(best_rect)
            box = np.intp(box)
            cv2.drawContours(frame_out, [box], 0, (0, 255, 0), 2)

            # Check if conditions to switch to tracker are met
            area = cv2.contourArea(box)
            if spotter_success_cycles > blur_success_thresh:
                blur_sigma = np.sqrt(area)
            if score > spotter_confidence_thresh and area > frame_width * frame_height / spotter_area_thresh:
                spotter_success_cycles += 1
                if spotter_success_cycles >= spotter_duration_thresh:
                    use_tracker = True
                    tracker_failures = 0  # Reset tracker failure count
        else:
            spotter_success_cycles = 0  # Reset success count

    end_time = time.time()

    # Display frame time in the top right corner
    frame_time = f"Time: {frame_number / frame_fps:.2f}s"
    ft_size, _ = cv2.getTextSize(frame_time, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    processing_time = end_time - start_time
    processing_display = f"Proc Time: {processing_time:.3f}s"
    fps_size, _ = cv2.getTextSize(processing_display, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_size = 0
    if fps_size > ft_size:
        text_size = fps_size
    else:
        text_size = ft_size
    text_x = frame_width - text_size[0] - 10  # Adjust X position
    text_y = 30  # Y position remains the same
    cv2.putText(frame_out, frame_time, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display processing time
    processing_time = end_time - start_time
    processing_display = f"Proc Time: {processing_time:.3f}s"
    cv2.putText(frame_out, processing_display, (text_x, text_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



    # Prepare FSM relevant information for display
    area_percentage = (area / (frame_width * frame_height) * 100) if best_rect_info else 0
    fsm_info = [
        f"Score: {score:.1f}" if best_rect_info else "Score: N/A",
        f"Success Cycles: {spotter_success_cycles}",
        f"Failures: {tracker_failures}",
        f"Area: {area_percentage:.1f}%" if best_rect_info else "Area: N/A",
        f"Using Homography Tracker: {use_tracker}"
    ]
    for i, line in enumerate(fsm_info):
        cv2.putText(frame_out, line, (10, frame_height - 100 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame_out)
    cv2.imshow('window', frame_out)

    if cv2.waitKey(1) == 27:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

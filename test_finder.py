import cv2
import argparse
from tracker import Tracker
from spotter import Spotter
from target_finder import TargetFinder

def main(input_video, reference_image, output_video=None):
    # Read the reference image
    ref_image = cv2.imread(reference_image)

    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize Tracker and Spotter
    tracker = Tracker(ref_image)
    spotter = Spotter(frame_width, frame_height)

    # Initialize TargetFinder with annotation always set to True
    target_finder = TargetFinder(tracker, spotter, frame_width, frame_height, annotate=True, sensor_dimensions=(0.006287,0.00353644))

    # VideoWriter for output, if specified
    out = None
    if output_video:
        frame_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, frame_fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        corners, annotated_frame = target_finder.process_frame(frame)

        # Display the annotated frame
        cv2.imshow('Annotated Frame', annotated_frame)

        # Write to output file if specified
        if out:
            out.write(annotated_frame)

        if cv2.waitKey(1) == 27:
            break

    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Target Finder Test Program')
    parser.add_argument('--input', help='Input video file', default='S1000644.LRV')
    parser.add_argument('--reference', help='Reference image file', default='qr_code.png')
    parser.add_argument('--output', help='Output video file', default='output.mp4')
    args = parser.parse_args()

    main(args.input, args.reference, args.output)

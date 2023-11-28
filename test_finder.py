import cv2
import argparse
from tracker import Tracker
from spotter import Spotter
from target_finder import TargetFinder

def main(input_source, reference_image, output_video=None):
    # Read the reference image
    ref_image = cv2.imread(reference_image)

    # Initialize variables
    use_camera = input_source.lower() == 'camera'
    frame_width, frame_height, frame_fps = None, None, None

    # Initialize the video source
    if use_camera:
        from pi_camera_stream import PiCameraStream
        # Set your desired resolution here
        camera_resolution = (1280, 720)
        camera_stream = PiCameraStream(resolution=camera_resolution)
        camera_stream.start_stream()
        frame_width, frame_height = camera_resolution
        frame_fps = 30  # Set this to your camera's framerate
    else:
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize Tracker, Spotter, and TargetFinder
    tracker = Tracker(ref_image)
    spotter = Spotter(frame_width, frame_height)
    target_finder = TargetFinder(tracker, spotter, frame_width, frame_height, annotate=True, sensor_dimensions=(0.006287,0.00353644))

    # VideoWriter for output, if specified
    out = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, frame_fps, (frame_width, frame_height))

    # Main processing loop
    while True:
        if use_camera:
            frame = camera_stream.read_frame()
            if frame is None:
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # Process the frame
        corners, annotated_frame = target_finder.process_frame(frame)
        #print(corners)

        # Display the annotated frame
        cv2.imshow('Annotated Frame', annotated_frame)

        # Write to output file if specified
        if out:
            out.write(annotated_frame)

        if cv2.waitKey(1) == 27:
            break

    # Release resources
    if use_camera:
        camera_stream.release()
    else:
        cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Target Finder Test Program')
    parser.add_argument('--input', help='Input source (video file or "camera")', default='S1000644.LRV')
    parser.add_argument('--reference', help='Reference image file', default='qr_code.png')
    parser.add_argument('--output', help='Output video file', default='output.mp4')
    args = parser.parse_args()

    main(args.input, args.reference, args.output)

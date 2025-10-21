import pickle
import cv2
import numpy as np
import sys
from pathlib import Path

def pickle_to_video(pickle_path, output_path="output.mp4", fps=15):
    """
    Reads a pickle file containing a list of dicts and creates a video
    from the 'observation.images.wrist_camera' frames.
    """

    # Load the pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a list of dicts in the pickle file.")

    frames = []
    for i, entry in enumerate(data):
        try:
            img = entry["observation.images.wrist_camera"]
            # Convert to uint8 numpy array if needed
            img = np.array(img, dtype=np.uint8)
            frames.append(img)
        except KeyError:
            print(f"Warning: Missing 'observation.images.wrist_camera' in entry {i}")
        except Exception as e:
            print(f"Error processing entry {i}: {e}")

    if not frames:
        raise ValueError("No valid frames found in the pickle file.")

    # Get frame size
    height, width = frames[0].shape[:2]
    print(f"Video resolution: {width}x{height}, Frames: {len(frames)}, FPS: {fps}")

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        # Ensure color format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
    print(f"✅ Video saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pickle_to_video.py <pickle_file> [output.mp4]")
        sys.exit(1)

    pickle_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

    pickle_to_video(pickle_file, output_file)
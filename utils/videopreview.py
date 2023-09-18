import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create an overlapped video from left and right image folders.")
    parser.add_argument('-lf', '--left_folder', type=str, required=True, help="Path to the left image folder")
    parser.add_argument('-rf', '--right_folder', type=str, required=True, help="Path to the right image folder")
    args = parser.parse_args()

    left_folder = args.left_folder
    right_folder = args.right_folder

    if not os.path.exists(left_folder) or not os.path.exists(right_folder):
        raise FileNotFoundError("One or both of the specified folders do not exist.")

    left_files = sorted([os.path.join(left_folder, file) for file in os.listdir(left_folder)])
    right_files = sorted([os.path.join(right_folder, file) for file in os.listdir(right_folder)])

    output_video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    for left_file, right_file in zip(left_files, right_files):
        left_frame = cv2.imread(left_file)
        right_frame = cv2.imread(right_file)

        left_frame = cv2.resize(left_frame, (640, 480))
        right_frame = cv2.resize(right_frame, (640, 480))

        overlapped_frame = left_frame.copy()

        opacity = 0.5

        cv2.addWeighted(right_frame, opacity, overlapped_frame, 1 - opacity, 0, overlapped_frame)

        cv2.imshow('Stereo Video', overlapped_frame)

        output_video.write(overlapped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

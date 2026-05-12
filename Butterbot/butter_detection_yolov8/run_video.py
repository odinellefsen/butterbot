import cv2
import argparse
from ultralytics import YOLO
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time object detection with YOLOv8 and dynamic webcam resolution.")
    parser.add_argument("--width", default=1280, type=int, help="Width of the webcam's resolution.")
    parser.add_argument("--height", default=720, type=int, help="Height of the webcam's resolution.")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # Initialize the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    # Load a custom model
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
    model = YOLO(model_path)
    threshold = 0.45  # Adjust the confidence threshold as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame")
            break

        # Detect objects in the frame
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                label = f"{results.names[int(class_id)].upper()} {score:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Detection', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



####################################################################################################################################################################################################
# import cv2
# from ultralytics import YOLO
# import os
# import argparse

# # Initialize the camera (0 by default, you may need to adjust this based on your camera setup)
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error opening video stream or file")
#     exit()

# # Load a model
# model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)  # Load a custom model
# threshold = 0.45  # Adjust the confidence threshold as needed

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab a frame")
#         break

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             label = f"{results.names[int(class_id)].upper()} {score:.2f}"  # Include confidence level in label
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, label, (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)  # Adjust text size if needed

#     # Display the resulting frame
#     cv2.imshow('Frame', frame)

#     # Press Q on keyboard to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


####################################################################################################################################################################################################

# import os
# import cv2
# from ultralytics import YOLO

# VIDEOS_DIR = os.path.join('.', 'videos')

# video_path = os.path.join(VIDEOS_DIR, 'IMG_2035.mp4')
# video_path_out = '{}_out.mp4'.format(video_path)

# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"Error opening video file {video_path}")
#     exit()

# ret, frame = cap.read()
# if not ret:
#     print("Failed to read the video")
#     cap.release()
#     exit()

# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# # Load a model
# model = YOLO(model_path)  # load a custom model

# threshold = 0.45  # Adjust the confidence threshold as needed

# while ret:
#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             label = f"{results.names[int(class_id)].upper()} {score:.2f}"  # Include confidence level in label
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, label, (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)  # Adjust text size if needed

#     out.write(frame)
#     ret, frame = cap.read()

# cap.release()
# out.release()
# cv2.destroyAllWindows()

####################################################################################################################################################################################################

# import cv2
# from ultralytics import YOLO

# video_path = 'videos/IMG_2035.mp4'

# # Load the video
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"Error opening video file {video_path}")
#     exit()

# # Load the model
# model_path = 'runs/detect/train/weights/last.pt'
# model = YOLO(model_path)  # Load a custom model
# threshold = 0.45  # Adjust the confidence threshold as needed

# while True:
#     # Read a video frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to read the video or video has ended")
#         break

#     # Perform object detection
#     results = model(frame)[0]

#     # Process detection results
#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             label = f"{results.names[int(class_id)].upper()} {score:.2f}"
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

#     # Display the frame
#     cv2.imshow('YOLOv8 Object Detection', frame)

#     # Break the loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

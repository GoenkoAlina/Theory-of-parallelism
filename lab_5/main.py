from ultralytics import YOLO
import threading
import queue
import time
import cv2
import argparse


point_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
               "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
               "right_knee", "left_ankle", "right_ankle"]

pairs = [(5, 7), (7, 9), (11, 13), (13, 15), (5, 11), (6, 12), (11, 12), (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7),
         (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]


def draw(frame, results):
    for i in range(len(results[0].keypoints.xy)):
        key_points = results[0].keypoints.xy[i]
        for start, end in pairs:
            x1, y1, x2, y2 = int(key_points[start][0]), int(key_points[start][1]), int(key_points[end][0]), int(
                key_points[end][1])
            if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
    return frame


class ThreadSafePredict:
    def __init__(self, frame_queue: queue.Queue, event_stop: threading.Event, frames):
        self._frame_queue = frame_queue
        self._event_stop = event_stop
        self._frames = frames
        self._model = YOLO(model="yolov8n-pose.pt")

    def get_result(self):
        while True:
            try:
                ind, frame = self._frame_queue.get(timeout=1)
                results = self._model.predict(source=frame, device='cpu')
                frame = draw(frame, results)
                frame = cv2.resize(frame, (640, 480))
                self._frames.append((ind, frame))
            except queue.Empty:
                if self._event_stop.is_set():
                    print(f'Thread {threading.get_ident()} final!')
                    break


class ThreadRead:
    def __init__(self, path_video: str, frame_queue: queue.Queue, event_stop: threading.Event):
        self._frame_queue = frame_queue
        self._event_stop = event_stop
        self._cap = cv2.VideoCapture(path_video)
        self._count = 0

    def read_frame(self):
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                print("Can't receive frame!")
                break
            self._frame_queue.put((self._count, frame))
            self._count += 1
            time.sleep(0.0001)
        self._cap.release()
        self._event_stop.set()


def command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='Path to the video')
    parser.add_argument('mode', type=str, help='Execution mode')
    parser.add_argument('output_name', type=str, help='The name of the output file')
    args = parser.parse_args()
    return args


def main():
    args = command_line_parser()

    if args.mode == 'single-threaded':
        count_threads = 1
    else:
        count_threads = 6

    threads = []
    frame_queue = queue.Queue(1000)
    event_stop = threading.Event()
    writer = cv2.VideoWriter(args.output_name, cv2.VideoWriter.fourcc(*'mp4v'), 10, (640, 480))
    frames = []

    thread_read = ThreadRead(args.video_path, frame_queue, event_stop)
    thread_read = threading.Thread(target=thread_read.read_frame)
    thread_read.start()

    start_t = time.monotonic()

    for _ in range(count_threads):
        thread_predict = ThreadSafePredict(frame_queue, event_stop, frames)
        threads.append(threading.Thread(target=thread_predict.get_result))

    for thr in threads:
        thr.start()

    for thr in threads:
        thr.join()

    thread_read.join()

    end_t = time.monotonic()
    print(f'Time: {end_t - start_t}')

    frames.sort()
    for key, value in frames:
        writer.write(value)
    writer.release()


main()

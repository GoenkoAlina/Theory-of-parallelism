import queue
import time
import cv2
import logging
import argparse
import threading
import numpy as np


q = queue.Queue()
q_cam = queue.Queue()
working = True


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam:
    def __init__(self, device, resolution):
        self._resolution = resolution
        self._frame = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
        try:
            self._cap = cv2.VideoCapture(device)
        except Exception as e:
            logging.error(str(e), exc_info=True)
            global working
            working = False

    def get(self):
        ret, self._frame = self._cap.read()
        if not ret:
            logging.error("Error receiving data from the camera", exc_info=True)
            global working
            working = False
        self._frame = cv2.resize(self._frame, self._resolution)
        return self._frame

    def __del__(self):
        self._cap.release()


class WindowImage:
    def __init__(self, frequency):
        self._frequency = frequency
        self._img = np.zeros((224, 224, 3), dtype=np.uint8)

    def show(self, img):
        self._img = img
        time.sleep(self._frequency)
        cv2.imshow('Frame', img)
        try:
            cv2.imshow('Frame', img)
        except Exception as e:
            logging.error(str(e), exc_info=True)
            global working
            working = False

    def __del__(self):
        cv2.destroyWindow('Frame')


def command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('device', type=int, help='Camera name')
    parser.add_argument('resolution_h', type=int, help='Camera resolution height')
    parser.add_argument('resolution_w', type=int, help='Camera resolution width')
    parser.add_argument('frequency', type=float, help='Frequency of the image display')
    args = parser.parse_args()
    return args


def setting_up_logging():
    logging.basicConfig(level=logging.INFO, filename="log/log.log", filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")


def work_thread(sensor_type, sensor):
    if sensor_type == 's_cam':
        queue_ = q_cam
    else:
        queue_ = q
    while working:
        response = sensor.get()
        queue_.put((sensor_type, response))


def main_thread(output, resolution_w, resolution_h):
    global working
    s0 = s1 = s2 = 0
    x0, y0 = resolution_w - 120, resolution_h - 60
    while working:
        _, image = q_cam.get()
        sensor_type, data = q.get()
        if sensor_type == 's0':
            s0 = data
        elif sensor_type == 's1':
            s1 = data
        else:
            s2 = data
        image = cv2.rectangle(image, (x0, y0), (resolution_w, resolution_h), (255, 255, 255), thickness=cv2.FILLED)
        image = cv2.putText(image, ('Sensor0: ' + str(s0)), (x0 + 3, resolution_h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1)
        image = cv2.putText(image, ('Sensor1: ' + str(s1)), (x0 + 3, resolution_h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1)
        image = cv2.putText(image, ('Sensor2: ' + str(s2)), (x0 + 3, resolution_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1)
        output.show(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            working = False
            break


def main():
    args = command_line_parser()

    setting_up_logging()

    sensor_cam = SensorCam(args.device, (args.resolution_w, args.resolution_h))
    sensor0 = SensorX(0.01)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(1)
    output = WindowImage(args.frequency)

    t_cam = threading.Thread(target=work_thread, args=('s_cam', sensor_cam))
    t0 = threading.Thread(target=work_thread, args=('s0', sensor0))
    t1 = threading.Thread(target=work_thread, args=('s1', sensor1))
    t2 = threading.Thread(target=work_thread, args=('s2', sensor2))

    t_cam.start()
    t0.start()
    t1.start()
    t2.start()
    main_thread(output, args.resolution_w, args.resolution_h)

    t_cam.join()
    t0.join()
    t1.join()
    t2.join()


main()


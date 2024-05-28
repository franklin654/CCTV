import signal
import sys
import time

import cv2
from ultralytics import YOLO

from VideoRecorder import VideoRecorder

font = cv2.FONT_HERSHEY_SIMPLEX


def sigterm_handler(_signo, _stack_frame):
    cap.release()
    del video_recorder
    cv2.destroyAllWindows()
    sys.exit(0)


def time_to_string():
    current_time = time.localtime()
    str_time = 'date: {}/{} time: {:02d}:{:02d}:{:02d}'.format(current_time.tm_mon, current_time.tm_mday,
                                                               current_time.tm_hour, current_time.tm_min,
                                                               current_time.tm_sec)
    return str_time


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_recorder = VideoRecorder(29.0, size)
    model = YOLO('C:\\Users\\saisi\\Memory_card\\python\\cctv\\yolov8s.pt')
    counter = 0
    try:
        while cap.isOpened():
            success, img = cap.read()
            text = time_to_string()
            if success:
                results = model(img, conf=0.65, classes=0, imgsz=img.shape[:2])
                annotated_frame = results[0].plot()
                cv2.putText(annotated_frame, text, (10, img.shape[0] - 20), font, 1, color=(255, 0, 0), thickness=2,
                            lineType=cv2.LINE_8, bottomLeftOrigin=False)
                if 0 in results[0].boxes.cls:
                    counter += 1
                    conf_level = int(results[0].boxes.conf[0].item() * 10)
                    currentTime = time.localtime()
                    row = {'confidence': str(conf_level),
                           'date': '{}-{}-{}'.format(currentTime.tm_mon,
                                                     currentTime.tm_mday,
                                                     currentTime.tm_year),
                           'time': '{:02d}:{:02d}:{:02d}'.format(currentTime.tm_hour,
                                                                 currentTime.tm_min,
                                                                 currentTime.tm_sec)}
                    video_recorder.write_row(row)
                cv2.imshow('STREAM 1', annotated_frame)
                video_recorder.write(annotated_frame)
                if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q')):
                    break
            else:
                break
        cap.release()
        del video_recorder
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print(KeyboardInterrupt)
        cap.release()
        del video_recorder
        cv2.destroyAllWindows()
        sys.exit(0)

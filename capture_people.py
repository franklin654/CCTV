import os.path
import signal
from ultralytics import YOLO
import cv2
import time
import sys

folder_name: str = ""


def sigterm_handler(_signo, _stack_frame):
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    sys.exit(0)


def dir_for_images():
    curr_time = time.localtime()
    global folder_name
    folder_name = f'G:\\My Drive\\cctv_footage\\captured_images_{curr_time.tm_mday}_{curr_time.tm_mon}_{curr_time.tm_year}'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


def prepare_video(frame_size: tuple):
    curr_time = time.localtime()
    video_name = f'G:\\My Drive\\cctv_footage\\captured_footage_{curr_time.tm_mday}_{curr_time.tm_mon}_{curr_time.tm_year}_{curr_time.tm_hour}_{curr_time.tm_min}_{curr_time.tm_sec}.avi'
    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    return cv2.VideoWriter(video_name, codec, 29.0, frame_size)


def time_to_string():
    current_time = time.localtime()
    str_time = 'date: {}/{} time: {:02d}:{:02d}:{:02d}'.format(current_time.tm_mon, current_time.tm_mday,
                                                               current_time.tm_hour, current_time.tm_min,
                                                               current_time.tm_sec)
    return str_time


def img_name():
    current_time = time.localtime()
    file_name = '{:02d}_{:02d}_{:02d}.png'.format(current_time.tm_hour, current_time.tm_min, current_time.tm_sec)
    return file_name


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)
    dir_for_images()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    video = prepare_video(size)
    model = YOLO('C:\\Users\\saisi\\Memory_card\\python\\cctv\\yolov8s.pt')
    counter = 0
    try:
        while cap.isOpened() and video.isOpened():
            success, img = cap.read()
            text = time_to_string()
            if success:
                results = model(img, conf=0.7, classes=0, imgsz=img.shape[:2])
                annotated_frame = results[0].plot()
                cv2.putText(annotated_frame, text, (10, img.shape[0]-20), font, 1, color=(255, 0, 0), thickness=2,
                            lineType=cv2.LINE_8, bottomLeftOrigin=False)
                if 0 in results[0].boxes.cls:
                    counter += 1
                    conf_level = int(results[0].boxes.conf[0].item()*10)
                    file_name = f'conf_{conf_level}_{img_name()}'
                    cv2.imwrite(folder_name+"\\"+file_name, annotated_frame)
                cv2.imshow('STREAM 1', annotated_frame)
                video.write(annotated_frame)
                if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q')):
                    break
            else:
                break
        cap.release()
        video.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print(KeyboardInterrupt)
        cap.release()
        video.release()
        cv2.destroyAllWindows()
        sys.exit(0)

import time
from typing import Union
import cv2
import csv


class VideoRecorder(cv2.VideoWriter):
    def __init__(self, frame_rate: float, frame_size: Union[list[int, int], tuple[int, int]]):
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self._generate_names()
        super().__init__(self._video_name, cv2.VideoWriter.fourcc(*"XVID"), self.frame_rate, frameSize=self.frame_size)
        self._csvfile = open(self.csv_filename, 'w', newline='')
        fieldnames = ['confidence', 'date', 'time']
        self._csvwriter = csv.DictWriter(self._csvfile, fieldnames)
        self._csvwriter.writeheader()

    def __del__(self):
        self._csvfile.close()
        super().release()

    def _generate_names(self):
        curr_time = time.localtime()
        self._video_name = "G:\\My Drive\\cctv_footage\\captured_footage_" \
                           "{}_{}_{}_{}_{}_{}.avi".format(curr_time.tm_mday,
                                                          curr_time.tm_mon,
                                                          curr_time.tm_year,
                                                          curr_time.tm_hour,
                                                          curr_time.tm_min,
                                                          curr_time.tm_sec)
        self.csv_filename = "G:\\My Drive\\cctv_footage\\captured_footage_" \
                            "{}_{}_{}_{}_{}_{}.csv".format(curr_time.tm_mday,
                                                           curr_time.tm_mon,
                                                           curr_time.tm_year,
                                                           curr_time.tm_hour,
                                                           curr_time.tm_min,
                                                           curr_time.tm_sec)

    def write_row(self, row: dict):
        self._csvwriter.writerow(row)

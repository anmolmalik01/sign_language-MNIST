# ===================================== dependencies =====================================
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

import os
import shutil

from hand_detection import handDetector
from directory import delete_directory, create_directory
from reshape import reshape

# ========================= deleting img folder and creating new img folder =========================

delete_directory('./images/hand_images')
delete_directory('./images/reshaped_images')
delete_directory('./images/video_to_images')
create_directory('./images/hand_images')
create_directory('./images/reshaped_images')
create_directory('./images/video_to_images')

# =================================== video to images ===================================

vidcap = cv2.VideoCapture('video.mp4')

# video to images function


def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite("./images/video_to_images/image"+str(count) +
                    ".jpg", image)     # save frame as JPG file
    return hasFrames


sec = 0
frameRate = 2.  # //it will capture image in each 0.5 second
count = 1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

print(f"Succesfully converted video to images at frames {frameRate} per/min")

# ========================= hand detector and croping hands in images =========================

x = handDetector()
x.main()

# # ====================================== reshaping images ======================================
reshape()

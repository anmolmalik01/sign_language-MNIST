import cv2
import os
import mediapipe as mp
import time
import matplotlib.pyplot as plt


# IMAGE_FILES = ['./backup/1.jpg']

# images
image_len = len(os.listdir('./images/video_to_images'))
IMAGE_FILES = []
for i in range(1, image_len+1):
    IMAGE_FILES.append('./images/video_to_images/image'+str(i)+'.jpg')


class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

    def findPosition(self, img, handNo=0, draw=False):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

    def main(self):
        count = 1
        print(len(IMAGE_FILES))

        with self.mpHands.Hands(static_image_mode=True, min_detection_confidence=0.5) as hands:
            for idx, file in enumerate(IMAGE_FILES):

                image = cv2.imread(file)
                self.results = hands.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # for hand number 0
                x = self.findPosition(image)
                print(x)
                if x:
                    smallest_x = x[0][1]
                    largest_x = x[0][1]
                    smallest_y = x[0][2]
                    largest_y = x[0][2]

                    for i in range(len(x)):
                        # samlest and largest x
                        if (x[i][1] < smallest_x):
                            smallest_x = x[i][1]
                        if (x[i][1] > largest_x):
                            largest_x = x[i][1]

                        # smallest and largest y
                        if (x[i][2] < smallest_y):
                            smallest_y = x[i][2]
                        if (x[i][2] > largest_y):
                            largest_y = x[i][2]

                    print(smallest_x)
                    print(largest_x)
                    print(smallest_y)
                    print(largest_y)

                    # for x
                    if(image.shape[0] - smallest_y-50) < 0:
                        crop_img = image[smallest_y: smallest_y + (
                            largest_y - smallest_y), smallest_x: smallest_x+(largest_x - smallest_x)]
                    if(image.shape[1] - smallest_x-50) < 0:
                        crop_img = image[smallest_y: smallest_y + (
                            largest_y - smallest_y), smallest_x: smallest_x+(largest_x - smallest_x)]
                    else:
                        crop_img = image[smallest_y-50: 50+smallest_y + (
                            largest_y - smallest_y), smallest_x-50: 50+smallest_x+(largest_x - smallest_x)]
                    cv2.imwrite("./images/hand_images/image" +
                                str(count)+".jpg", crop_img)
                    count = count + 1

        print("Succesfully detected hands and saved images")


# x = handDetector()
# x.main()

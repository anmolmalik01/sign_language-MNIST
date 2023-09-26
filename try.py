# Importing Image from PIL package
import cv2 
# creating a image object
im = cv2.imread("./backup/3.jpg")
print(im)
print('=======================')
op = cv2.imread('./backup/3.jpg')
print(op)
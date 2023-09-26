from PIL import Image
import os

# IMAGE_FILES = ['./backup/2.jpg']

# IMAGE_FILES = os.listdir('hh.jpg')
# images
image_len = len(os.listdir('./images/hand_images'))
IMAGE_FILES = []
for i in range(1, image_len+1):
    IMAGE_FILES.append('./images/hand_images/image'+str(i)+'.jpg')


def reshape():
    count = 1
    for i in range(len(IMAGE_FILES)):
        img = Image.open(IMAGE_FILES[i]).convert('L')
        resized_img = img.resize((28, 28))
        resized_img.save("./images/reshaped_images/image"+str(count)+".jpg")
        count = count + 1

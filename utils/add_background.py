import os
import cv2
import random
import math

path = 'data/images'
list_file = os.listdir(os.path.expanduser(path))
try:
    list_file.remove('__pycache__')
except:
    pass

x = 1

choice = [1, 1.1, 1.2, 1.5, 2]
choice_background = ['b1.jpg', 'b2.jpg', 'b3.jpg', 'b4.jpg', 'b5.jpg']
for file in list_file:
    background = cv2.imread(f'data/background/{random.choice(choice_background)}')
    background = cv2.resize(background, (224 * 3, 320 * 3))
    h, w = background.shape[:2]
    img = cv2.imread(f'{path}/{file}')
    scale = random.choice(choice)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    h1, w1 = img.shape[:2]
    w2 = (w - w1) // 2
    h2 = (h - h1) // 2
    print(x)
    background[h2:h2 + h1, w2:w2 + w1] = img

    cv2.imwrite(f'data/images_face/img_b_face_{x}.jpg', background)
    x += 1

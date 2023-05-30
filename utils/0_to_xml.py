import os
import dlib
from utils.auto_label.image_to_xml import label_to_xml
import cv2

detector = dlib.get_frontal_face_detector()
path = 'data/images_face'


def box_face(frame):
    boxes = []
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector(imgRGB, 0)
    for bbox in results:
        x1 = bbox.left()
        y1 = bbox.top()
        x2 = bbox.right()
        y2 = bbox.bottom()
        boxes = [x1, y1, x2, y2]
        cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        break
    return boxes, frame.shape[:2], frame


def crop(img):
    # print(img.shape)

    img_crop = img[:, 120:520, :]

    return img_crop


cap = cv2.VideoCapture(0)
count = 1
x = 701
while count <= 100:

    ret, frame = cap.read()
    file = f'class_face_{x}.jpg'
    frame = crop(frame)
    cv2.imwrite(f'{path}/{file}', frame)
    box, shape, frame = box_face(frame)
    boxes = [box, [], [], []]
    if len(boxes[0]) == len(boxes[1]) == len(boxes[2]) == len(boxes[3]) == 0:
        print(f'đã xoá {file}')
        os.remove(f'{path}/{file}')
    else:
        label_to_xml('auto_label/form_detect.xml', boxes, file,
                     f"data/labels/{file.rstrip('.jpg')}.xml", shape)
        count += 1
        x += 1

    print(count)
    cv2.imshow('window', frame)
    cv2.waitKey(250)

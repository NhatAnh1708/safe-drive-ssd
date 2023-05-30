import os
import dlib
from utils.auto_label.image_to_xml import label_to_xml
import cv2

detector = dlib.get_frontal_face_detector()
path = 'data/images_face'
list_file = os.listdir(os.path.expanduser(path))
try:
    list_file.remove('__pycache__')
except:
    pass


def box_face(path):
    boxes = []
    frame = cv2.imread(path)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector(imgRGB, 0)
    for bbox in results:
        x1 = bbox.left()
        y1 = bbox.top()
        x2 = bbox.right()
        y2 = bbox.bottom()
        boxes = [x1, y1, x2, y2]
    return boxes, frame.shape[:2]


for file in list_file:
    box, shape = box_face(f'{path}/{file}')
    boxes = [box, [], [], []]
    if len(boxes[0]) == len(boxes[1]) == len(boxes[2]) == len(boxes[3]) == 0:
        print(f'đã xoá {file}')
        os.remove(f'{path}/{file}')
    else:
        label_to_xml('auto_label/form_detect.xml', boxes, file,
                     f"data/labels/{file.rstrip('.jpg')}.xml", shape)

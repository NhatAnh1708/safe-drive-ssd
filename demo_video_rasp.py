import cv2

from library.detect_face import detect_face
from library.drowsiness import drow

from onnx_model import predict_ONNX
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
import time
# Load model
import numpy as np
from datetime import datetime
import math
# predict
model = predict_ONNX('export_model/onnx_model/model.onnx')
category_index = label_map_util.create_category_index_from_labelmap("data/label_map.txt", use_display_name=True)


def demo_video(path_video):
    cap = cv2.VideoCapture(path_video)


    def check_drow(img, output_dict):
        result_drowsiness = False
        boxes = detect_face(img, output_dict, thresding=0.5)

        if len(boxes) == 4:
            img, result_drowsiness = drow(img, boxes, 0.15)

        return result_drowsiness


    def safe_drive(img, output_dict, thresding=0.5):
        cd = check_drow(img, output_dict)
        for i in range(4):
            if (output_dict['detection_classes'][i] == 2 and output_dict['detection_scores'][i] >= thresding) \
                    or (output_dict['detection_classes'][i] == 3 and output_dict['detection_scores'][i] >= thresding) \
                    or cd:
                return False
        for i in range(4):
            if output_dict['detection_classes'][i] == 4 and not cd \
                    and output_dict['detection_scores'][i] >= thresding:
            # if not cd:
                return True
        return False


    def equa_hist(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    count_safe = 0

    count = 0
    while (cap.isOpened()):
        count += 1
        t = time.time()
        ret, frame = cap.read()
        # print(type(frame))
        if not ret:
            break
        image_np = frame
        # image_np = equa_hist(image_np)
        # print("Done load image ")
        img_ed = np.expand_dims(image_np, axis=0)

        output_dict = model.predict_model(img_ed)
        image_np = cv2.resize(image_np, dsize=(224, 320), fx=2, fy=2)

        # print("Done inference")
        boxes = detect_face(image_np, output_dict, thresding=0.5)
        check = safe_drive(image_np, output_dict, 0.5)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            min_score_thresh=.5,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=3)
        w, h = image_np.shape[:2]

        if not check:
            count_safe += 1
        else:
            count_safe = 0
        if count_safe >= 1:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            cv2.putText(image_np, f'{dt_string}: safety attention ', (10,  int(0.75*w)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        try:
            fps = 1 / (time.time() - t)
            cv2.putText(image_np, f'fps: {str(math.ceil(fps))}', (int(0.75*h), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except:
            pass

        cv2.imshow('window', image_np)
        cv2.waitKey(1)
    print('Done!')
    cap.release()

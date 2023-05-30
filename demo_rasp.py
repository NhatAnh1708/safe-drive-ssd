import threading

from library.lib import *
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
import time
from picamera2 import Picamera2
from onnx_model import predict_ONNX
import numpy as np
from datetime import datetime
import math


class system:
    def __init__(self):
        self.model = predict_ONNX('export_model/onnx_model/model.onnx')
        self.category_index = label_map_util.create_category_index_from_labelmap("data/label_map.txt",
                                                                                 use_display_name=True)
        self.img = None
        self.count_drowsiness = 0
        self.alarmed = False
        self.result_drowsiness = False

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        self.picam2.start()
        threading.Thread(target=self.run_video).start()
        threading.Thread(target=self.check_safe).start()
        self.count_safe = 0

    def equa_hist(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def run_video(self, scale=1):
        count = 0
        while True:
            frame = self.picam2.capture_array()

            def crop(img):
                img_crop = img[:, 120:520, :3]

                return img_crop
            if count % 3 == 0:
                image_np = self.equa_hist(crop(frame))
                image_np = cv2.resize(image_np, dsize=(224, 320), fx=scale, fy=scale)
                self.img = image_np

    def check_drow(self, img, output_dict):
        result_drowsiness = False
        boxes = detect_face(img, output_dict, thresding=0.5)

        if len(boxes) == 4:
            img, result_drowsiness = drow(img, boxes, 0.3)

        return result_drowsiness

    def safe_drive(self, img, output_dict, thresding=0.5):
        cd = self.check_drow(img, output_dict)
        for i in range(4):
            if (output_dict['detection_classes'][i] == 2 and output_dict['detection_scores'][i] >= thresding) \
                    or (output_dict['detection_classes'][i] == 3 and output_dict['detection_scores'][i] >= thresding) \
                    or cd:
                return False
        for i in range(4):
            # if output_dict['detection_classes'][i] == 4 and not cd \
            #         and output_dict['detection_scores'][i] >= thresding:
            if not cd:
                return True
        return False

    def check_safe(self):
        print('check')
        while self.img is None:
            time.sleep(0.1)
        print('check1')
        while True:
            t = time.time()

            img = self.img.copy()
            img_ed = np.expand_dims(img, axis=0)

            output_dict = self.model.predict_model(img_ed)

            boxes = detect_face(img, output_dict, thresding=0.4)

            if len(boxes) == 4:
                img, self.result_drowsiness = drow(img, boxes)

            if self.result_drowsiness:
                self.count_drowsiness += 1
            else:
                self.count_drowsiness = 0
            print(self.count_drowsiness)
            vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                output_dict['detection_boxes'][:5],
                output_dict['detection_classes'][:5],
                output_dict['detection_scores'][:5],
                self.category_index,
                min_score_thresh=.4,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=3)
            check = self.safe_drive(img, output_dict, 0.5)

            if not check:
                self.count_safe += 1
            else:
                self.count_safe = 0
            try:
                fps = 1 / (time.time() - t)
                cv2.putText(img, f'fps: {str(math.ceil(fps))}', (120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if self.count_safe >= int(fps * 2):
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    cv2.putText(img, f'{dt_string}: safety attention ', (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            except:
                pass

            cv2.imshow('window', img)
            cv2.waitKey(1)


if __name__ == '__main__':
    cv2.startWindowThread()
    safe = system()

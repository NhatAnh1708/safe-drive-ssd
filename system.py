import threading
from library.lib import *
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
import time
from onnx_model import predict_ONNX
import numpy as np
from datetime import datetime
import math


class system:
    def __init__(self, video):
        self.model = predict_ONNX('export_model/onnx_model/model.onnx')

        self.category_index = label_map_util.create_category_index_from_labelmap("data/label_map.txt",
                                                                                 use_display_name=True)
        self.img = None
        self.count_drowsiness = 0
        self.alarmed = False
        self.result_drowsiness = False
        self.cap = cv2.VideoCapture(video)
        self.total_video = 0
        self.total_predict = 0
        self.count_video = 0
        self.count_predict = 0
        self.count_safe = 0
        # self.stop_signal = False
        self.count = 0
        self.video_thread = threading.Thread(target=self.run_video)
        self.proces_thread = threading.Thread(target=self.check_safe)
        self.video_thread.start()
        self.proces_thread.start()
        self.video_thread.join()
        self.proces_thread.join()

    def crop(self, img):
        return img[:, 120:520, :]

    def equa_hist(self, img):

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def run_video(self, scale=1):
        self.count = 0
        while self.cap.isOpened():
            self.count += 1
            self.count_video += 1
            t = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.count % 3 == 0:
                image_np = self.equa_hist(self.crop(frame[:, ::-1]))
                image_np = cv2.resize(image_np, dsize=(224,320), fx=scale, fy=scale)
                image_np = cv2.resize(image_np, dsize=None, fx=2, fy=2)

                self.img = image_np
                # time.sleep(0.001)
            else:
                continue
            self.total_video += (time.time() - t)
            # if self.stop_signal:
            #     print('run_video: ', self.total_video)
            #     break
            # time.sleep(0.1)

    def check_drow(self, img, output_dict):
        result_drowsiness = False
        boxes = detect_face(img, output_dict, thresding=0.6)

        if len(boxes) == 4:
            img, result_drowsiness = drow(img, boxes, 0.3)

        return result_drowsiness
    def safe_drive(self, img, output_dict, thresding=0.6):
        cd = self.check_drow(img, output_dict)
        # print('...',cd)
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
            self.count_predict += 1
            t = time.time()

            img = self.img.copy()
            img_ed = np.expand_dims(img, axis=0)

            output_dict = self.model.predict_model(img_ed)

            check = self.safe_drive(img,output_dict,0.6)
            # print(check)
            vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                min_score_thresh=.6,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=3)

            if not check:
                self.count_safe += 1
            else:
                self.count_safe = 0
            try:
                fps = 1 / (time.time() - t)
                cv2.putText(img, f'fps: {str(math.ceil(fps))}', (150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if self.count_safe >= int(fps * 2):
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    cv2.putText(img, f'{dt_string}: safety attention ', (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except:
                pass

            cv2.imshow('window', img)
            cv2.waitKey(1)


if __name__ == '__main__':
    safe = system(0)

from library.lib import *
from models.research.object_detection.utils import label_map_util
from onnx_model import predict_ONNX
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

class eval_matrix:
    def __init__(self):
        self.model = predict_ONNX('export_model/onnx_model/model.onnx')
        self.category_index = label_map_util.create_category_index_from_labelmap("data/label_map.txt",
                                                                                 use_display_name=True)
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def equa_hist(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def predict(self, image):
        # image_np = self.equa_hist(image)
        image_np = image
        img_ed = np.expand_dims(image_np, axis=0)

        output_dict = self.model.predict_model(img_ed)
        return output_dict

    def check_drow(self, img, output_dict):
        result_drowsiness = False
        boxes = detect_face(img, output_dict, thresding=0.5)

        if len(boxes) == 4:
            img, result_drowsiness = drow(img, boxes, 0.15)

        return result_drowsiness

    def safe_drive(self, img, output_dict, thresding=0.5):
        cd = self.check_drow(img, output_dict)
        # print('...',cd)
        for i in range(4):
            if (output_dict['detection_classes'][i] == 2 and output_dict['detection_scores'][i] >= thresding) \
                    or (output_dict['detection_classes'][i] == 3 and output_dict['detection_scores'][i] >= thresding) \
                    or cd:
                return False
        for i in range(4):
            if output_dict['detection_classes'][i] == 4 and not cd \
                    and output_dict['detection_scores'][i] >= thresding:
                return True
        return False

    def print_confusion_matrix(self, confusion_matrix, class_names, figsize=(6, 6), fontsize=25):
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
        )
        fig = plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        fig.savefig(os.path.join("confusion_matrix.png"))
        plt.show()

    def eval(self, path):
        True_predict = 0
        False_predict = 0
        files = os.listdir(os.path.expanduser(f'{path}'))
        try:
            files.remove('__pycache__')
        except:
            pass
        # print(len(files))
        for file in tqdm(files):
            img = cv2.imread(f'{path}/{file}')
            img = cv2.resize(img, dsize=(224, 320), fx=2, fy=2)

            output_dict = self.predict(img)
            # print(file)
            # print(output_dict['detection_classes'][:4])
            # print(output_dict['detection_scores'][:4])
            check = self.safe_drive(img, output_dict)
            # print(check)
            if check:
                True_predict += 1
            else:
                False_predict += 1
        return True_predict, False_predict


if __name__ == '__main__':
    cfs_matrix = eval_matrix()
    TP, FP = cfs_matrix.eval('../eval_dataset/0')
    # # print('-----------------------')
    FN, TN = cfs_matrix.eval('../eval_dataset/1')
    # print(1250, 384, 147, 1466)
    c = np.array([[TP, FP], [FN, TN]])
    # print(c)
    # TP, FP, FN,TN = (1250, 384, 147, 1466)
    cfs_matrix.print_confusion_matrix(c, ['0', '1'])
    acc = (TP + TN) / (FP + FN + TP + TN) * 100
    pre_P = TP/(TP+FP)*100
    pre_N = TN/(TN+FN)*100
    recall_P = TP/(TP+FN)*100
    recall_N = TN/(TN+FP)*100
    F1_score = 2*(pre_P*recall_P)/(pre_P+recall_P)
    print('Accuracy: %f' % acc)
    print('Precision_P: %f' % pre_P)
    print('Precision_N: %f' % pre_N)

    print('Recall_P: %f' % recall_P)
    print('Recall_N: %f' % recall_N)

    print('F1 score: %f' % F1_score)



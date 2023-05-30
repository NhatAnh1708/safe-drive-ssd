from library.lib import *
from models.research.object_detection.utils import label_map_util
import numpy as np
from onnx_model import predict_ONNX
import os
from utils.auto_label.image_to_xml import label_to_xml
from tqdm import tqdm
# Load model
model = predict_ONNX('../export_model/onnx_model/model.onnx')

# predict

category_index = label_map_util.create_category_index_from_labelmap("../data/label_map.txt", use_display_name=True)

path = '/home/datdt/DATN/eval_metric_mAP/images'
files = os.listdir(os.path.expanduser(path))
try:
    files.remove('__pycache__')
except:
    pass
for file in tqdm(files):
    image_np = cv2.imread(f'{path}/{file}')

    img_ed = np.expand_dims(image_np, axis=0)

    output_dict = model.predict_model(img_ed)
    box_face = detect_face(image_np, output_dict, 0.5)
    box_phone = detect_phone(image_np, output_dict, 0.5)
    box_drinks = detect_drinks(image_np, output_dict, 0.5)
    box_seatbelt = detect_seatbelt(image_np, output_dict, 0.5)
    boxes = [box_face, box_phone, box_drinks, box_seatbelt]

    label_to_xml('auto_label/form_detect.xml', boxes, file,
                 f"labels_predict/{file.rstrip('.jpg')}.xml", image_np.shape[:2])

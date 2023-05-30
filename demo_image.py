import cv2
from IPython.display import display
from library.inference import load_image_into_numpy_array, run_inference_for_single_image

from PIL import Image
from library.detect_face import detect_face
import tensorflow as tf
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
import numpy as np
from onnx_model import predict_ONNX
# Load model


# predict

category_index = label_map_util.create_category_index_from_labelmap("data/label_map.txt", use_display_name=True)

# image_path = '/content/drive/MyDrive/Safe-Drive/data/split_data/test/img_1413.jpg'
image_path = 'data/test/augmentaton_seatbelt_30.jpg'
tf.keras.backend.clear_session()
model = tf.saved_model.load("export_model/saved_model")
image_np = load_image_into_numpy_array(image_path)
print("Done load image ")
image_np = cv2.resize(image_np, dsize=(224,224), fx=1, fy=1)

# image_np = cv2.resize(image_np, dsize=None, fx=1, fy=1)
output_dict = run_inference_for_single_image(model, image_np)
print("Done inference")
# vis_util.visualize_boxes_and_labels_on_image_array(
#     image_np,
#     output_dict['detection_boxes'],
#     output_dict['detection_classes'],
#     output_dict['detection_scores'],
#     category_index,
#     min_score_thresh=.5,
#     instance_masks=output_dict.get('detection_masks_reframed', None),
#     use_normalized_coordinates=True,
#     line_thickness=3)
# for i in range(len(output_dict['detection_classes'])):
#   if output_dict['detection_classes'][i] == 2:
#     print(output_dict['detection_boxes'][i], ':',i)

print("Done draw on image ")
# display(Image.fromarray(image_np))
# cv2.imshow('window',image_np)
# cv2.waitKey(0)
print(output_dict['raw_detection_boxes'].shape)

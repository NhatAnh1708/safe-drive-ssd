import numpy as np
import onnxruntime
from models.research.object_detection.utils import ops as utils_ops
import tensorflow as tf


class predict_ONNX:
    def __init__(self,
                 model_path: str,
                 providers: list = None):
        self.model_path = model_path
        self.input_shape = [112, 112]
        if providers is None:
            self.providers = ['CUDAExecutionProvider',
                              'CPUExecutionProvider', ]
        else:
            self.providers = providers
        self.sess = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
        print(self.sess.get_providers())

    def run(self, image: np.ndarray):
        return self.sess.run(None, {'input_tensor': image})

    def dictionary_output(self, output_list):
        dicts = {
            'raw_detection_scores': output_list[7],
            'detection_boxes': output_list[1],
            'detection_multiclass_scores': output_list[3],
            'detection_anchor_indices': output_list[0],
            'detection_scores': output_list[4],
            'detection_classes': output_list[2],
            'raw_detection_boxes': output_list[6],
            'num_detections': output_list[5]
        }
        # print(dicts)
        return dicts
    def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width, resize_method='bilinear'):
        """Transforms the box masks back to full image masks.

        Embeds masks in bounding boxes of larger masks whose shapes correspond to
        image shape.

        Args:
            box_masks: A tensor of size [num_masks, mask_height, mask_width].
            boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
                corners. Row i contains [ymin, xmin, ymax, xmax] of the box
                corresponding to mask i. Note that the box corners are in
                normalized coordinates.
            image_height: Image height. The output mask will have the same height as
                        the image height.
            image_width: Image width. The output mask will have the same width as the
                        image width.
            resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
            'bilinear' is only respected if box_masks is a float.

        Returns:
            A tensor of size [num_masks, image_height, image_width] with the same dtype
            as `box_masks`.
        """
        resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
        # TODO(rathodv): Make this a public function.
    def predict_model(self, image):
        output_dict = self.run(image)
        output_dict = self.dictionary_output(output_dict)
        num_detections = int(output_dict.pop('num_detections')) - 96
        # print(num_detections)
        output_dict = {key: value[0, :num_detections]
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        if 'detection_masks' in output_dict:
            detection_masks_reframed = reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])

            detection_masks_reframed = np.array(detection_masks_reframed > 0.5, dtype=np.uint8)
            # dtype = np.uint8
            output_dict['detection_masks_reframed'] = detection_masks_reframed
        # print(output_dict)
        return output_dict

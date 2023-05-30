def detect_drinks(image, output_dict,thresding = 0.5, check = True):
    boxes = []
    im_height, im_width = image.shape[:2]

    if check:
        def toado(boxes):
            (left, right, top, bottom) = (round(boxes[1] * im_width), round(boxes[3] * im_width),
                                          round(boxes[0] * im_height), round(boxes[2] * im_height))
            return left, right, top, bottom
    else:
        def toado(boxes):
            (left, right, top, bottom) = ((boxes[1]), (boxes[3]),
                                          (boxes[0]), (boxes[2]))
            return left, right, top, bottom

    try:

        for i in range(10):
            if output_dict['detection_classes'][i] == 3 and output_dict['detection_scores'][i] >= thresding:
                boxes = list(output_dict['detection_boxes'][i])
                break
    except:
        pass
    # print(boxes)
    if len(boxes) == 4:
        left, right, top, bottom = toado(boxes)
        # print(f'{left}, {right}, {top}, {bottom}')
        boxes = [left, top, right, bottom]
    return boxes

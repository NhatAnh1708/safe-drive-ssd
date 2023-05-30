import cv2 as cv
import numpy as np


class ImageAst:
    def __init__(self, img_path):

        self.window_name = 'show_img'
        # self.img = cv.imread(img_path)
        self.img = img_path

        self.count = 0
        self.list_mouse = [[102, 3], [556, 0], [636, 476], [9, 468]]

    def on_mouse(self, *args):

        if args[0] == cv.EVENT_LBUTTONDOWN:
            x = args[1]
            y = args[2]
            self.list_mouse.append([x,y])
            self.count +=1
            print(f'{x}', f',{y}', self.img[y, x])

    def show(self, window_name, image, is_wait=False):
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, self.on_mouse)
        cv.imshow(window_name, image)
        if is_wait:
            cv.waitKey(0)

    def transform(self):
        if self.count >= 4:
            points_one = np.array([self.list_mouse[0], self.list_mouse[1], self.list_mouse[2], self.list_mouse[3]], dtype=np.float32)
            list_a_0 = []
            list_a_1 = []
            for x in self.list_mouse:
                list_a_0.append(x[0])
                list_a_1.append(x[1])
            print(list_a_0)
            w, h = (max(list_a_0)-min(list_a_0)), (max(list_a_1)-min(list_a_1))
            points_two = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            M = cv.getPerspectiveTransform(points_one, points_two)
            ret_img = cv.warpPerspective(self.img, M, (w, h))
            # print(ret_img.shape)
            # self.show(window_name='ret', image=ret_img, is_wait=True)
            # max x - min x, max y - min y
            print(ret_img)
            return ret_img


if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    while True:
        ret,frame = cap.read()
    # ia = ImageAst(img_path='data/test/img_959.jpg')
        ia = ImageAst(img_path=frame)
        ia.show(window_name='Image', image=ia.img, is_wait=True)
        ia.transform()
        print(ia.list_mouse)
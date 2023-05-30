from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
# from playsound import playsound
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
print("[INFO] loading facial landmark predictor...")
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def drow(frame, boxes, EYE_AR_THRESH=0.25):
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    output_image = frame.copy()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[2]
    y2 = boxes[3]
    rect = dlib.rectangle(x1, y1, x2, y2)
    shape = predictor(gray, rect)
    # print('shape', shape, type(shape))
    shape = face_utils.shape_to_np(shape)
    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    COUNTER = 0
    if ear < EYE_AR_THRESH:

        # print(COUNTER)
        # if COUNTER > EYE_AR_CONSEC_FRAMES:
        cv2.putText(frame, "day day!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        results = True

    # otherwise, the eye aspect ratio is not below the blink
    # threshold, so reset the counter and alarm
    else:

        ALARM_ON = False
        results = False
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame, results


if __name__ == '__main__':
    frame = cv2.imread('img_288.jpg')
    boxes = [64, 87, 119, 159]
    frame = drow(frame, boxes)

    cv2.imshow("Frame", frame)
    cv2.waitKey(0) & 0xFF
    # do a bit of cleanup
    cv2.destroyAllWindows()
    frame.stop()

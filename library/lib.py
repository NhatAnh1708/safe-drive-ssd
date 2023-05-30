import cv2

#from library.inference import run_inference_for_single_image
from library.detect_face import detect_face
from library.detect_phone import detect_phone
from library.detect_drinks import detect_drinks
from library.detect_seatbelt import detect_seatbelt
from library.drowsiness import drow
from library.playsound_wav import play_sound_wav
from threading import Thread
#import tensorflow as tf

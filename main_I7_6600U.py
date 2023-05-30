import argparse
from demo_video import demo_video
from system import system
parser = argparse.ArgumentParser('main_I7_6600U.py -input video/camera(0)')
parser.add_argument('-input', '--input', default='0', help='Video/camera(0) ')
args = parser.parse_args()
input_path = args.input

if input_path == '0':
    system(0)
else:
    demo_video(input_path)


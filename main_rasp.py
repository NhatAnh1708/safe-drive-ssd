import argparse
from demo_video_rasp import demo_video
from system import system
from demo_pi_rasp import system_rasp
parser = argparse.ArgumentParser('main_rasp.py -input video/camera(0)/camera_pi(pi)')
parser.add_argument('-input', '--input', default='pi', help='path_Video/camera(0)/camera_pi(pi)')
args = parser.parse_args()
input_path = args.input

if input_path == '0':
    system(0)
elif input_path == 'pi':
    system_rasp()
else:
    demo_video(input_path)



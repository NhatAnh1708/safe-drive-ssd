import os
from threading import Thread
from playsound import playsound
# wav_path =

def play_sound_wav():
    playsound('alarm.wav')
if __name__ == '__main__':
    alarmed = False
    try:
        if not alarmed:
            alarmed = True
            # Duong dan den file wav

            # Tien hanh phat am thanh trong 1 luong rieng
            t = Thread(target=play_sound_wav,
                       args=())
            t.deamon = True
            t.start()
    except:
        pass


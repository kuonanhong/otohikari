import cv2
import numpy as np
from rgb_streaming import PixelCatcher, mouseParam

import matplotlib.pyplot as plt


def video_2_frames(video_file, x=0, y=0, h=1, w=1):

    cap = cv2.VideoCapture(video_file)  # Video to frames

    print(x,y,h,w)

    img = PixelCatcher([[x,y]],[h,w])
    while(cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False:  # Is a frame left?stats
            break

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img.process(frame)

    cap.release()  # When everything done, release the capture
    return img.extract()



def xy_pixel_definition(image_file):
    read = cv2.imread(image_file)
    window_name = "input window"
    cv2.imshow(window_name, read)

    mouseData = mouseParam(window_name)

    while 1:
        cv2.waitKey(20)
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN: #left click
            y,x = mouseData.getPos()
        elif mouseData.getEvent() == cv2.EVENT_LBUTTONDBLCLK: #left W-click
            break;

    cv2.destroyAllWindows()
    return x, y



if __name__ == '__main__':

    video_file = 'video/20180426_led_calibration_1_cut.avi'
    image_dir = 'image/20180426_led_calibration_1/'
    image_file = "image/20180426_led_calibration_1/pixel_accuss.png"

    fps = 30.0003

    x, y = xy_pixel_definition(image_file)
    h = 20
    w = 20

    values = video_2_frames(video_file, x, y, h, w)

    len = len(values[:,...])
    t = [i/fps for i in range(len)]

    #plt.figure(figsize=(16,4))

    R = plt.plot(t, values[...,0], label = 'Red')
    G = plt.plot(t, values[...,1], label = 'Green')
    B = plt.plot(t, values[...,2], label = 'Blue')
    plt.title('RGB POWER')
    plt.xlabel('time [sec]')
    plt.ylabel('power')
    plt.legend()
    plt.xlim(0, t[-1])
    plt.show()

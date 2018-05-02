import os
import shutil
import cv2
import numpy as np
from streaming import OnlineStats

import matplotlib.pyplot as plt


def video_2_frames(video_file=' ', image_dir=' ', image_file='img_%s.png'):
    # Video to frames
    i = 0
    cap = cv2.VideoCapture(video_file)
    #frame_count = int(cap.get(7))
    #frame_cut = np.zeros(frame_count, h, w)
    stats = None
    x = 322
    y = 632
    h = 16
    w = 18
    while(cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False:  # Is a frame left?stats
            break
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #cv2.imwrite(image_dir+image_file % str(i).zfill(6), frame)  # Save a frame
        #print('Save', image_dir+image_file % str(i).zfill(6))

        frame_cut = frame[x:x+h,y:y+w]
        #if processor is not None:
        #    processor.process(frame_cut[None,:,:,:])
        cv2.imwrite(image_dir+'cut/'+image_file % str(i).zfill(6), frame_cut)
        i += 1


    cap.release()  # When everything done, release the capture
    return i



def frame_ave():

    import glob
    from PIL import Image

    files = glob.glob('image/20180426_led_calibration_1/cut/*.png')

    frame_pixel_r_ave = np.array([])
    frame_pixel_g_ave = np.array([])
    frame_pixel_b_ave = np.array([])
    for file in files:
        pixel_r_sum = 0
        pixel_g_sum = 0
        pixel_b_sum = 0

        img = Image.open(file).convert('RGB')
        w, h = img.size

        pixel_count = 0
        for y in range(h):
            for x in range(w):
                r, g, b = img.getpixel((x, y))
                pixel_r_sum += r
                pixel_g_sum += g
                pixel_b_sum += b
                pixel_count += 1

        frame_pixel_r_ave = np.append(frame_pixel_r_ave, pixel_r_sum / pixel_count )
        frame_pixel_g_ave = np.append(frame_pixel_g_ave, pixel_g_sum / pixel_count )
        frame_pixel_b_ave = np.append(frame_pixel_b_ave, pixel_b_sum / pixel_count )
    return frame_pixel_r_ave, frame_pixel_g_ave, frame_pixel_b_ave




if __name__ == '__main__':

    video_file = './video/20180426_led_calibration_1_cut.avi'
    image_dir = './image/20180426_led_calibration_1/'

    f = 30.0003

    #processor = OnlineStats(3)

    frame_num = video_2_frames(video_file, image_dir)

    r_ave = np.array([])
    g_ave = np.array([])
    b_ave = np.array([])
    r_ave, g_ave, b_ave = frame_ave()


    x = range(frame_num)

    plt.plot(x, r_ave)
    plt.plot(x, g_ave)
    plt.plot(x, b_ave)
    plt.xlabel('frame number')   ###frame number to time
    plt.ylabel('level')
    plt.show()

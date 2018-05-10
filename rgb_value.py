import cv2
import numpy as np
from rgb_streaming import PixelCatcher, mouseParam
from NMF import NMF
import pyroomacoustics as pra
from pyroomacoustics import stft

import matplotlib.pyplot as plt
import json



def video_2_frames(video_file, x=0, y=0, h=1, w=1, start=0, end=13410):
    print('x:[',x-h//2,x+h//2,'] y:[',y-w//2,y+w//2,']')
    cap = cv2.VideoCapture(video_file)  # Video to frames
    cap.set(1, start)

    img = PixelCatcher([[x,y]],[h,w])
    count = 0
    while(cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False or count == end-start:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img.process(frame)
        count += 1

    cap.release()  # When everything done, release the capture
    return img.extract()



def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / ( x_max - x_min)
    return x_norm



def least_square_fitting(x, y, dim, color):
    x = np.array(x)
    x = min_max_normalization(x)
    y = min_max_normalization(y)
    c = np.polyfit(x, y, dim)
    fit_curve = np.sum(x[:,None]**np.arange(dim+1)*c[::-1].T, axis=1)
    print('y = {0}x^4+{1}x^3+{2}x^2+{3}x+{4}'.format(c[0],c[1],c[2],c[3],c[4]))

    plt.plot(x, fit_curve, label='fitting curve')
    plt.plot(x, y, label=color)
    plt.xlim(0, x[-1])
    plt.legend()
    plt.show()



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
    c = "green"
    f_dim = 4
    f = open("parameter.json", "r")
    data = json.load(f)

    x, y = xy_pixel_definition(data[c]["image_file"])

    values = video_2_frames(data[c]["video_file"], x, y,
                            data[c]["h"], data[c]["w"],
                            data[c]["start"], data[c]["end"])

    len = len(values[:,...])
    t = [i/data[c]["fps"] for i in range(len)]

    #least_square_fitting(t, values[...,data[c]["color_num"]], f_dim, c)

    frlen = 1024
    frsht = frlen // 2
    win = pra.hann(frlen)

    s_values = stft.stft(values[...,0], frlen, frsht, win=win)

    '''

    R = 50
    n_iter = 300

    nmf_value = NMF(values[...,data[c]["color_num"]], R=R, n_iter=n_iter)

    for i in range(3):
        print(i, np.shape(nmf_value[i]))

    a = librosa.istft(nmf_db[1][0,:] * (np.cos(np.angle(S_db) + 1j * np.sin(np.angle(S_db)))))
    b = librosa.istft(nmf_db[1][1,:] * (np.cos(np.angle(S_db) + 1j * np.sin(np.angle(S_db)))))

    plt.subplot(211)
    plt.plot(remixed_d)
    plt.subplot(212)
    plt.plot(remixed_b)
    plt.tight_layout()
    plt.show()
''
    R = plt.plot(t, values[...,0], label = 'Red')
    G = plt.plot(t, values[...,1], label = 'Green')
    B = plt.plot(t, values[...,2], label = 'Blue')
    plt.title('RGB POWER')
    plt.xlabel('time [sec]')
    plt.ylabel('power')
    plt.legend()
    plt.xlim(0, t[-1])
    plt.show()
'''

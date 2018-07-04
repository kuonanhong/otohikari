import argparse, json, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from readers import ThreadedVideoStream, ProcessorBase

def ccw3p(p1, p2, img_size):
    '''
    Finds all the points that are anti-clock wise (or colinear)
    wrt to p1 and p2 in an image of img_size

    :arg p1: (ndarray size 2) coordinates of a 2D point
    :arg p2: (ndarray size 2) coordinates of a 2D point
    :arg img_size: (ndarray size 2) coordinates of a 2D point

    :returns: (int) orientation of the given triangle
        1 if triangle vertices are counter-clockwise
        -1 if triangle vertices are clockwise
        0 if vertices are collinear

    :ref: https://en.wikipedia.org/wiki/Curve_orientation
    '''
    Y, X = np.meshgrid(range(img_size[1]), range(img_size[0]))

    p1, p2 = np.array(p1), np.array(p2)

    if p1.shape[0] != 2 or p2.shape[0] != 2:
        raise ValueError('geometry.ccw3p is for three 2D points')

    d = (p2[0] - p1[0]) * (Y - p1[1]) - (X - p1[0]) * (p2[1] - p1[1])
    ret = np.ones(img_size, dtype=np.bool)
    ret[d < -1e-5] = 0

    return ret


class Tracker(ProcessorBase):

    def __init__(self, protocol, red_thresh, bg_len, search_box, monitor=False, qlen=10):

        ProcessorBase.__init__(self, monitor=monitor, qlen=qlen)

        self.protocol = protocol
        self.red_thresh = red_thresh
        self.bg_len = bg_len
        self.search_box = search_box

        self.bg_set = []
        self.background = None
        self.buffer_frame = None
        self.mask = None

        self.is_tracking = False
        self.current_location = None
        self.val = [-1,-1,-1]

        self.f_counter = 0
        self.f_shape = None

        self.trajectory = []

    def make_mask(self):

        if self.f_shape is None:
            raise ValuelError('The frame shape must be set when calling make_mask')

        # we need to create a second small mask for the
        # reflection of the blinkies on the ground
        p_lo = self.protocol['blinky_reflection_offset_param']['lo']
        p_hi = self.protocol['blinky_reflection_offset_param']['hi']
        def ref_interp(y, p_lo, p_hi):
            c = (y - p_hi[0]) / (p_lo[0] - p_hi[0])
            return c * (p_lo[1] - p_hi[1]) + p_hi[1]

        # Create a mask of blinky locations
        blinky_mask = 255 * np.ones(self.f_shape, dtype=np.uint8)
        for blinky in self.protocol['blinky_locations']:
            y,x = blinky
            self.mask = cv2.circle(blinky_mask, (x,y), 4, (0,0,0), -1)
            # add the reflection
            y_r = int(np.round(y + ref_interp(y, p_lo, p_hi)))
            self.mask = cv2.circle(blinky_mask, (x,y_r), 4, (0,0,0), -1)

        # create a mask to catch the floor only
        floor_mask = 255 * np.ones(self.f_shape, dtype=np.uint8)
        for lbl, pm in protocol['floor_mask_param'].items():
            p1,p2 = pm
            m = ccw3p(p1, p2, floor_mask.shape[:2])
            floor_mask *= np.array(1 - m, dtype=np.uint8)[:,:,None]

        self.mask = cv2.bitwise_and(blinky_mask, floor_mask)

    def __process__(self, frame):

        self.f_counter += 1

        if self.f_shape is None:
            self.f_shape = frame.shape
            self.background = np.zeros_like(frame)
            self.buffer_frame = np.zeros_like(frame)
            self.make_mask()

        if self.f_counter < self.bg_len:
            self.bg_set.append(frame)

        elif self.f_counter == self.bg_len:
            self.background[:,:,:] = np.mean(self.bg_set, axis=0)

        else:

            # determinen search area
            if self.current_location is None:
                ylo = 0
                xlo = 0
                # no idea where to look...
                sx = slice(None)
                sy = slice(None)
            else:
                # look in a box around previous location
                ylo = self.current_location[0] - self.search_box // 2
                if ylo < 0:
                    ylo = 0
                yhi = self.current_location[0] + self.search_box // 2
                sy = slice(
                        ylo,
                        yhi if yhi <= self.f_shape[0] else self.f_shape[0]
                        )
                xlo = self.current_location[1] - self.search_box // 2
                if xlo < 0:
                    xlo = 0
                xhi = self.current_location[1] + self.search_box // 2
                sx = slice(
                        xlo,
                        xhi if xhi <= self.f_shape[1] else self.f_shape[1]
                        )

            # background subtraction, mask, thresholding
            self.buffer_frame[:,:,:] = 0
            self.buffer_frame[sy,sx,:] = np.maximum(0, frame[sy,sx,:].astype(np.float) - self.background[sy,sx,:])
            self.buffer_frame[sy,sx,:] = cv2.bitwise_and(self.buffer_frame[sy,sx,:], self.mask[sy,sx,:])
            reddish = self.buffer_frame[sy,sx,2]

            # find max intensity red pixel
            flat_ind = np.argmax(reddish.ravel())
            y, x = np.unravel_index(flat_ind, reddish.shape)
            self.val = reddish[y,x]
            y += ylo
            x += xlo

            # detect with threshold
            if self.val > self.red_thresh:
                self.current_location = [y, x]
                self.trajectory.append([self.f_counter, self.current_location])
            else:
                self.current_location = None

    def get_locations(self):
        return self.trajectory



if __name__ == '__main__':

    video_choices = [ 'noise', 'speech', 'hori_1', 'hori_2', 'hori_3', 'hori_4', 'hori_5', ]

    parser = argparse.ArgumentParser(description='Extract locations of blinkies and moving source from vide')
    parser.add_argument('protocol', type=str,
            help='The protocol file containing the experiment metadata')
    parser.add_argument('-v', '--video', type=str, choices=video_choices,
            help='The video segement to process')
    parser.add_argument('-s', '--show', action='store_true',
            help='Show the video')
    parser.add_argument('-m', '--mask', action='store_true',
            help='Display the masked video')
    args = parser.parse_args()

    # get the path to the experiment files
    experiment_path = os.path.split(args.protocol)[0]

    # Read some parameters from the protocol
    with open(args.protocol,'r') as f:
        protocol = json.load(f)

    # read in the important stuff
    blinkies = np.array(protocol['blinky_locations'])
    video_path = os.path.join(experiment_path, protocol['videos'][args.video])

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    tracker = Tracker(protocol, red_thresh=100, bg_len=400, search_box=300, monitor=True)

    with ThreadedVideoStream(video_path) as cap:

        cap.start()

        while cap.is_streaming():

            frame = cap.read()
            tracker(frame)

            if args.show:

                if tracker.current_location is not None:
                    y, x = tracker.current_location
                    frame = cv2.circle(frame, (x,y), 5, color[0].tolist(), 1)

                cv2.putText(frame, str(tracker.val),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

            if args.mask:
                cv2.putText(reddish, str(val),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('with mask', reddish)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    cv2.destroyAllWindows()

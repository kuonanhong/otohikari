import numpy as np
import cv2

class OnlineStats(object):
    '''
    Compute statistics on the input data in an online way

    Parameters
    ----------
    shape: tuple of int
        Shape of a data point tensor

    Attributes
    ----------
    mean: array_like (shape)
        Mean of all input samples
    var: array_like (shape)
        Variance of all input samples
    count: int
        Sample size
    '''

    def __init__(self, ndim):
        '''
        Initialize everything to zero
        '''
        self.ndim = ndim
        self.shape = None

    def process(self, data):
        '''
        Update statistics with new data

        Parameters
        ----------
        data: array_like
            A collection of new data points of the correct shape in an array
            where the first dimension goes along the data points
        '''
        if self.shape is None:
            self.shape = data.shape[-self.ndim:]
            self.mean = np.zeros(self.shape, dtype=np.float64)
            self.var = np.zeros(self.shape, dtype=np.float64)
            self.count = 0

        if data.shape[-len(self.shape):] != self.shape:
            raise ValueError('The data.shape[1:] should match the statistics object shape')

        data.reshape((-1,) + self.shape)

        count = data.shape[0]
        mean = np.mean(data, axis=0)
        var = np.var(data, axis=0)

        m1 = self.var * (self.count)

        m2 = var * (count)

        M2 = m1 + m2 + (self.mean - mean) ** 2 * count * self.count / (count + self.count)

        self.mean = (count * mean + self.count * self.mean) / (count + self.count)
        self.count += count
        self.var = M2 / (self.count)

class PixelCatcher(object):
    '''
    This is a simple object that collect the values of a few pixels on the stream of frames

    Parameters
    ----------
    pixels: list of tuples
        The location of the pixels to collect in the image
    '''
    def __init__(self, pixels, box_size):
        self.pixels = pixels
        self.values = []
        self.box_size = box_size

    def process(self, frames):
        '''
        Catch the values of the pixels in a stack of frames

        Parameters
        ----------
        frames: array_like (n_frames, height, width, 3)
            The stack of frames
        '''

        vals = [frames[loc[0]-self.box_size[0]//2:loc[0]+self.box_size[0]//2,
                        loc[1]-self.box_size[1]//2:loc[1]+self.box_size[1]//2,:].reshape((-1, frames.shape[2]))
                        for loc in self.pixels]
        vals = np.mean(vals, axis=1, keepdims = True)

        self.values += [vals]

    def extract(self):
        '''
        Format the values captured into an (n_frames, n_pixels, 3) shaped array
        '''
        v = np.concatenate(self.values, axis=0)
        return v.reshape((-1, len(self.pixels), 3))


class mouseParam(object):
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):

        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent

    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])

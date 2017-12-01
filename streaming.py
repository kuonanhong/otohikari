import numpy as np

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

    def __init__(self, shape):
        '''
        Initialize everything to zero
        '''
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.zeros(shape, dtype=np.float64)
        self.count = 0

    def process(self, data):
        '''
        Update statistics with new data

        Parameters
        ----------
        data: array_like
            A collection of new data points of the correct shape in an array
            where the first dimension goes along the data points
        '''

        if data.shape[-len(self.shape):] != self.shape:
            raise ValueError('The data.shape[1:] should match the statistics object shape')

        data.reshape((-1,) + self.shape)

        count = data.shape[0]
        mean = np.mean(data, axis=0)
        var = np.var(data, axis=0)

        m1 = self.var * (self.count - 1)
        m2 = var * (count - 1)
        M2 = m1 + m2 + (self.mean - mean) ** 2 * count * self.count / (count + self.count)

        self.mean = (count * mean + self.count * self.mean) / (count + self.count)
        self.count += count
        self.var = M2 / (self.count - 1)

class PixelCatcher(object):
    '''
    This is a simple object that collect the values of a few pixels on the stream of frames

    Parameters
    ----------
    pixels: list of tuples
        The location of the pixels to collect in the image
    '''
    def __init__(self, pixels):
        self.pixels = pixels
        self.values = []

    def process(self, frames):
        '''
        Catch the values of the pixels in a stack of frames

        Parameters
        ----------
        frames: array_like (n_frames, height, width, 3)
            The stack of frames
        '''

        vals = [frames[:,loc[0],loc[1],None,:] for loc in self.pixels]
        vals = np.concatenate(vals, axis=1)

        self.values += [vals]

    def extract(self):
        '''
        Format the values captured into an (n_frames, n_pixels, 3) shaped array
        '''
        v = np.concatenate(self.values, axis=0)
        return v.reshape((-1, len(self.pixels), 3))


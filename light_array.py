
import pyroomacoustics as pra
import samplerate
import numpy as np

class LightArray(pra.MicrophoneArray):
    '''
    This is an abstraction for the sound to light sensors
    we will be using.
    '''
    def __init__(self, R, fs):
        pra.MicrophoneArray.__init__(self, R, fs)

    def record(self, signals, fs):
        '''
        This simulates the recording of the signals by the microphones
        and the transformation from sound to light.
         
        Parameters
        ----------

        signals:
            An ndarray with as many lines as there are microphones.
        fs:
            the sampling frequency of the signals.
        '''

        # running average of power
        pwr = np.diff(signals) ** 2
        #pwr = signals ** 2

        if fs != self.fs:
            ratio = self.fs / fs

            # libsamplerate is limited to ratio in [1/256, 256] it seems
            # proceed in multiple stages
            counter = 0
            while ratio < (1 / 256):
                ratio *= 2
                counter += 1

            signals = pwr.T
            
            signals = samplerate.resample(signals, ratio, 'sinc_best')
            
            for c in range(counter):
                signals = samplerate.resample(signals, 0.5, 'sinc_best')

            self.signals = signals.T

        else:
            self.signals = pwr

        self.signals = 10 * np.log10(np.abs(self.signals))


class LightArray2(np.ndarray):

    def __new__(cls, input_array, fs=None, signals=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.fs = fs
        obj.signals = signals
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.fs = getattr(obj, 'fs', None)
        self.signals = getattr(obj, 'signals', None)

    def record(self, signals, fs):
        '''
        This simulates the recording of the signals by the microphones
        and the transformation from sound to light.
         
        Parameters
        ----------

        signals:
            An ndarray with as many lines as there are microphones.
        fs:
            the sampling frequency of the signals.
        '''

        # running average of power
        pwr = np.diff(signals) ** 2

        if fs != self.fs:
            ratio = self.fs / fs

            # libsamplerate is limited to ratio in [1/256, 256] it seems
            # proceed in multiple stages
            counter = 0
            while ratio < (1 / 256):
                ratio *= 2
                counter += 1

            signals = pwr.T
            
            signals = samplerate.resample(signals, ratio, 'sinc_best')
            
            for c in range(counter):
                signals = samplerate.resample(signals, 0.5, 'sinc_best')

            self.signals = signals.T

        else:
            self.signals = pwr

        self.signals = 10 * np.log10(np.abs(self.signals))


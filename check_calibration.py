import numpy as np
import matplotlib.pyplot as plt

def check_calibration(video, video_time, audio, audio_time, stats):

    # That was for a specific test
    '''
    pos = dict(
            green=np.s_[:,406:409,594:597,:],
            red=np.s_[:,411:414,594:597,:],
            blue=np.s_[:,406:409,600:603,:],
            white=np.s_[:,411:414,600:603,:],
            )
    '''

    # find the pixel with most variance
    y, x, ch = np.unravel_index(np.argmax(stats.var), stats.var.shape)
    pos = dict(
            maximum=np.s_[:,y:y+1,x:x+1,:],
            )

    for color, arr_slice in pos.items():

        # average a piece of image
        light_signal = video[arr_slice].mean(axis=1).mean(axis=1)

        from cycler import cycler
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

        '''
        plt.figure()
        plt.plot(video_time, light_signal)
        plt.title('LED color: ' + color)
        plt.savefig('led_signal_' + color + '.pdf')
        '''

        # now plot
        plt.figure()

        plt.subplot(2,1,1)
        plt.plot(video_time, light_signal)

        plt.subplot(2,1,2)
        plt.plot(audio_time, audio)

        plt.figure()

        audio = audio - np.mean(audio, axis=0)[None,:]

        # running average of power
        block = int(44100 / 30)
        pwr = (audio[:-(audio.shape[0] % block),0] ** 2).reshape((-1, block)).mean(axis=1)
        fs_pwr = 44100 / block
        pwr_time = np.arange(pwr.shape[0]) / fs_pwr

        light_signal -= light_signal.min()

        pwr_db = 10 * np.log10(pwr)
        pwr_db -= pwr_db[6:].min()

        pwr_db *= light_signal[:,2].max() / pwr_db.max()
        plt.plot(video_time, light_signal[:,2], '--')
        plt.plot(pwr_time, pwr_db, '-')
        plt.ylim([0, 1.05 * np.max(pwr_db)])

        plt.show()




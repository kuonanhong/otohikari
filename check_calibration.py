import numpy as np
import matplotlib.pyplot as plt

def check_calibration(video, video_time, audio, audio_time, stats):
    # find the pixel with most variance
    y, x, ch = np.unravel_index(np.argmax(stats.var), stats.var.shape)

    from cycler import cycler
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    # now plot
    plt.figure()

    light_signal = video[:,y,x,:]
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




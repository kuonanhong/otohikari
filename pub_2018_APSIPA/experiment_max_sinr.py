'''
This file applies a max SINR approach using the VAD information
from the LED and the two channels from the camera

Author: Robin Scheibler
Created: 2017/12/01
'''
import argparse, os, json, sys
import numpy as np
import scipy.linalg as la
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from mir_eval.separation import bss_eval_images
from tools import get_git_hash

import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_palette('pastel')

#import numpy.fft as fft
import mkl_fft as fft

from max_sinr_beamforming import compute_variances, compute_gain

experiment_folder = '../measurements/20180523/'
protocol_file = os.path.join(experiment_folder, 'session-201805{session}/protocol.json')
metadata_file = os.path.join(experiment_folder, 'session-201805{session}/processed/metadata.json')

blinky_sig = 'blinky_red'  # use the red channel
fs = 16000
thresh_opt = { -5 : 0.73, 0 : 0.73, 5 : 0.73, 10 : 0.73, 15 : 0.73, 20 : 0.73 }  # 2018/05/23

target_choices = ['ch' + str(i+1) for i in range(4)]
sir_choices = [-5, 0, 5, 10, 15, 20]
mic_choices = {
        'camera' : 'camera',
        'pyramic_4' : 'pyramic',
        'pyramic_24' : 'pyramic',
        'pyramic_48' : 'pyramic',
        }
blinky_source_map = dict(zip(target_choices, list(range(len(target_choices)))))

# subsets of pyramic microphones to use for BSS
pyramic_bss_4ch = [0, 16, 32, 15]
pyramic_bss_4ch = [8, 31, 24, 15]
pyramic_bss_4ch = [8, 9, 30, 31]

def fast_corr(signal, template):
    ''' correlation with fft '''
    return fftconvolve(signal, template[::-1], mode='same')

def process_experiment_max_sinr(SIR, mic, blinky, args):

    session = args.session
    target = args.target

    with open(metadata_file.format(session=args.session), 'r') as f:
        metadata = json.load(f)

    file_pattern = os.path.join(experiment_folder, metadata['filename_pattern'])

    with open(protocol_file.format(session=args.session), 'r') as f:
        protocol = json.load(f)

    nfft = args.nfft
    vad_guard = args.vad_guard
    if args.thresh is None:
        vad_thresh = thresh_opt[SIR]
    else:
        vad_thresh = args.thresh

    # read_in the mix signals
    fs_led, leds   = wavfile.read(file_pattern.format(
        session=session, snr=SIR, mic=blinky, source='mix', fs=fs))
    fs_snd, audio  = wavfile.read(file_pattern.format(
        session=session, snr=SIR, mic=mic_choices[mic], source='mix', fs=fs))
    assert fs_led == fs_snd

    # read in the ref signals
    sources_ref  = dict(zip(target_choices,
        [ wavfile.read(file_pattern.format(
                session=session, mic=mic_choices[mic], snr=SIR, source=ch, fs=fs))[1]
            for ch in target_choices ]))
    leds_ref  = dict(zip(target_choices,
        [ wavfile.read(file_pattern.format(
                session=session, mic=blinky, snr=SIR, source=ch, fs=fs))[1]
            for ch in target_choices ]))
    speech_ref  = sources_ref[target]

    noise_ref = np.zeros_like(sources_ref[target])
    n_ch = [ch for ch in target_choices if ch != target]
    for ch in n_ch:
        noise_ref += sources_ref[ch]

    # In case of objective evaluation, we do an artificial mix
    if args.synth_mix:
        audio = sources_ref[target] + noise_ref

    # get the geometry information to get nice plots.
    mics_geom = {
            'pyramic' : np.array(protocol['geometry']['microphones']['pyramic']['locations']),
            'camera'  : np.array(protocol['geometry']['microphones']['camera']['locations']),
            }

    mics_loc = np.array(protocol['geometry']['microphones'][mic_choices[mic]]['reference'])
    noise_loc = protocol['geometry']['speakers']['locations'][0]
    speech_loc = protocol['geometry']['speakers']['locations'][1]

    # the directions of arrival
    theta_speech = 0
    p0 = speech_loc - mics_loc
    p1 = noise_loc - mics_loc
    theta_noise = np.arccos(np.inner(p0, p1) / la.norm(p0) / la.norm(p1))
    print('Source separation', theta_noise / np.pi * 180)

    if 'pyramic' in mic:

        if mic == 'pyramic_4':
            I = pyramic_bss_4ch
        elif mic == 'pyramic_24':
            I = list(range(8,16)) + list(range(24,32)) + list(range(40,48)) # flat part
        else:
            I = list(range(48))
        I_bss = [I.index(i) for i in pyramic_bss_4ch]
            
        audio = audio[:,I]
        noise_ref = noise_ref[:,I].copy()
        speech_ref = speech_ref[:,I].copy()
        mics_positions = mics_geom['pyramic'][I].copy()
        # place in room 2-806
        mics_positions -= np.mean(mics_positions, axis=0)[None,:]
        mics_positions[:,2] -= np.max(mics_positions[:,2])
        mics_positions += mics_loc

        for ch in sources_ref:
            sources_ref[ch] = sources_ref[ch][:,I].copy()

    elif mic == 'camera':
        mics_positions = mics_geom['camera'].copy() + mics_loc


    n_samples = audio.shape[0]  # shorthand
    n_channels = audio.shape[1]

    # adjust length of led signal if necessary
    if leds.shape[0] < audio.shape[0]:
        z_missing = audio.shape[0] - leds.shape[0]
        leds = np.pad(leds, (0,z_missing), 'constant')
    elif leds.shape[0] > audio.shape[0]:
        leds = leds[:audio.shape[0],]

    # perform VAD
    led_target = leds[:,blinky_source_map[target]]
    vad_snd = led_target > vad_thresh

    # Now we want to make sure no speech speech goes in estimation of the noise covariance matrix.
    # For that we will remove frames neighbouring the detected speech
    vad_guarded = vad_snd.copy()
    if vad_guard is not None:
        for i,v in enumerate(vad_snd):
            if np.any(vad_snd[i-vad_guard:i+vad_guard]):
                vad_guarded[i] = True

    ##############################
    ## STFT and frame-level VAD ##
    ##############################

    print('STFT and stuff')
    sys.stdout.flush()

    a_win = pra.hann(nfft)
    s_win = pra.realtime.compute_synthesis_window(a_win, nfft // 2)

    engine = pra.realtime.STFT(nfft, nfft // 2,
            analysis_window=a_win, synthesis_window=s_win,
            channels=audio.shape[1])

    # Now compute the STFT of the microphone input
    X = engine.analysis(audio)
    X_time = np.arange(1, X.shape[0]+1) * (nfft / 2) / fs_snd

    X_speech = engine.analysis(audio * vad_guarded[:audio.shape[0],None])
    X_noise = engine.analysis(audio * (1 - vad_guarded[:audio.shape[0],None]))

    S_ref = engine.analysis(speech_ref)
    N_ref = engine.analysis(noise_ref)

    ##########################
    ## MAX SINR BEAMFORMING ##
    ##########################

    print('Max SINR beamformer computation')
    sys.stdout.flush()

    # covariance matrices from noisy signal
    Rall = np.einsum('i...j,i...k->...jk', X, np.conj(X))
    Rs = np.einsum('i...j,i...k->...jk', X_speech, np.conj(X_speech))
    Rn = np.einsum('i...j,i...k->...jk', X_noise, np.conj(X_noise)) 

    # compute covariances with reference signals to check everything is working correctly
    #Rs = np.einsum('i...j,i...k->...jk', S_ref, np.conj(S_ref))
    #Rn = np.einsum('i...j,i...k->...jk', N_ref, np.conj(N_ref))

    # compute the MaxSINR beamformer
    w = [la.eigh(rs, b=rn, eigvals=(n_channels-1,n_channels-1))[1] for rs,rn in zip(Rall[1:], Rn[1:])]
    w = np.squeeze(np.array(w))
    nw = la.norm(w, axis=1)
    w[nw > 1e-10,:] /= nw[nw > 1e-10,None]
    w = np.concatenate([np.ones((1,n_channels)), w], axis=0)

    if not args.no_norm:
        # normalize with respect to input signal
        z = compute_gain(w, X_speech, X_speech[:,:,0], clip_up=args.clip_gain)
        w *= z[:,None]


    ###########
    ## APPLY ##
    ###########

    print('Apply beamformer')
    sys.stdout.flush()

    # 2D beamformer
    mic_array = pra.Beamformer(mics_positions[:,:2].T, fs=fs_snd, N=nfft, hop=nfft, zpb=nfft)
    mic_array.signals = audio.T
    mic_array.weights = w.T

    out = mic_array.process()

    # Signal alignment step
    ref = np.vstack([speech_ref[:,0], noise_ref[:,0]])

    # Not sure why the delay is sometimes negative here... Need to check more
    delay = np.abs(int(pra.tdoa(out, speech_ref[:,0].astype(np.float), phat=True)))
    if delay > 0:
        out_trunc = out[delay:delay+ref.shape[1]]
        noise_eval = audio[:ref.shape[1],0] - out_trunc
    else:
        out_trunc = np.concatenate((np.zeros(-delay), out[:ref.shape[1]+delay]))
        noise_eval = audio[:ref.shape[1],0] - out_trunc
    sig_eval = np.vstack([out_trunc, noise_eval])

    # We use the BSS eval toolbox
    metric = bss_eval_images(ref[:,:,None], sig_eval[:,:,None])

    # we are only interested in SDR and SIR for the speech source
    ret = { 'Max-SINR' : {'SDR' : metric[0][0], 'SIR' : metric[2][0]} }

    #############################
    ## BLIND SOURCE SEPARATION ##
    #############################

    if mic in ['camera', 'pyramic_4']:

        if 'pyramic' in mic:
            X_bss = X[:,:,I_bss]
            ref = np.vstack(
                    [speech_ref[:,I_bss[0]]] + [sources_ref[ch][:,I_bss[0]]
                        for ch in target_choices if ch != target])
        elif mic == 'camera':
            X_bss = X

        Y = pra.bss.auxiva(X_bss, n_iter=40)
        bss = pra.realtime.synthesis(Y, nfft, nfft // 2, win=s_win)

        match = []
        for col in range(bss.shape[1]):
            xcorr = fast_corr(bss[:,col], ref[0])
            match.append(np.max(xcorr))
        best_col = np.argmax(match)

        # Not sure why the delay is sometimes negative here... Need to check more
        delay = np.abs(int(pra.tdoa(bss[:,best_col], speech_ref[:,0].astype(np.float), phat=True)))
        if delay > 0:
            bss_trunc = bss[delay:delay+ref.shape[1],]
        elif delay < 0:
            bss_trunc = np.concatenate((np.zeros((-delay, bss.shape[1])), bss[:ref.shape[1]+delay]))
        else:
            bss_trunc = bss[:ref.shape[1],]

        if ref.shape[1] > bss_trunc.shape[0]:
            ref_lim = bss_trunc.shape[0]
        else:
            ref_lim = ref.shape[1]

        metric = bss_eval_images(ref[:,:ref_lim,None], bss_trunc.T[:,:,None])
        SDR_bss = metric[0][0]
        SIR_bss = metric[2][0]
        ret['BSS'] = { 'SDR' : metric[0][0], 'SIR' : metric[2][0] }

    ##################
    ## SAVE SAMPLES ##
    ##################

    if args.save_sample is not None:

        if not os.path.exists(args.save_sample):
            os.makedirs(args.save_sample)

        # for informal listening tests, we need to high pass and normalize the
        # amplitude.
        if mic in ['camera', 'pyramic_4']:
            upper = np.max([audio[:,0].max(), out.max(), bss.max()])
        else:
            upper = np.max([audio[:,0].max(), out.max()])
        sig_in = pra.highpass(audio[:,0].astype(np.float) / upper, fs_snd, fc=150)
        sig_out = pra.highpass(out / upper, fs_snd, fc=150)

        f1 = os.path.join(args.save_sample, '{}_ch0_SIR_{}_dB.wav'.format(mic, SIR))
        wavfile.write(f1, fs_snd, sig_in)
        f2 = os.path.join(args.save_sample, '{}_maxsinr_SIR_{}_dB.wav'.format(mic, SIR))
        wavfile.write(f2, fs_snd, sig_out)

        if mic in ['camera', 'pyramic_4']:
            sig_bss = pra.highpass(bss[:,best_col] / upper, fs_snd, fc=150)
            f3 = os.path.join(args.save_sample, '{}_bss_SIR_{}_dB.wav'.format(mic, SIR))
            wavfile.write(f3, fs_snd, sig_bss)


    ##########
    ## PLOT ##
    ##########

    if args.plot:

        plt.figure()
        plt.plot(out_trunc)
        plt.plot(speech_ref[:,0])
        plt.legend(['output', 'reference'])

        # time axis for plotting
        led_time = np.arange(led_target.shape[0]) / fs_led + 1 / (2 * fs_led)
        audio_time = np.arange(n_samples) / fs_snd

        plt.figure()
        plt.plot(led_time, led_target, 'r')
        plt.title('LED signal')

        # match the scales of VAD and light to sound before plotting
        q_vad = np.max(audio)
        q_led = np.max(audio) / np.max(led_target)

        plt.figure()
        plt.plot(audio_time, audio[:,0], 'b') 
        plt.plot(led_time, led_target * q_led, 'r')
        plt.plot(audio_time, vad_snd * q_vad, 'g')
        plt.plot(audio_time, vad_guarded * q_vad, 'g--')
        plt.legend(['audio','VAD'])
        plt.title('LED and audio signals')

        plt.figure()
        a_time = np.arange(audio.shape[0]) / fs_snd
        plt.plot(a_time, audio[:,0])
        plt.plot(a_time, out_trunc)
        #plt.plot(a_time, speech_ref[:,0])
        plt.legend(['channel 0', 'beamformer output', 'speech reference'])

        '''
        plt.figure()
        mic_array.plot_beam_response()
        plt.vlines([180+np.degrees(theta_speech), 180-np.degrees(theta_noise)], 0, nfft // 2)

        room = pra.ShoeBox(protocol['geometry']['room'][:2], fs=16000, max_order=1)

        room.add_source(noise_loc[:2])   # noise
        room.add_source(speech_loc[:2])  # speech
        room.add_source(protocol['geometry']['speakers']['locations'][1][:2])  # signal

        room.add_microphone_array(mic_array)
        room.plot(img_order=1, freq=[800, 1000, 1200, 1400, 1600, 2500, 4000])
        '''

        plt.figure()
        mic_array.plot()

        plt.show()


    # Return SDR and SIR
    return ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('SIR', nargs='?', type=int, choices=sir_choices, help='The SIR between speech and noise')
    parser.add_argument('mic', nargs='?', type=str, choices=mic_choices.keys(), help='Which input device to use')
    parser.add_argument('target', type=str, choices=target_choices, help='The target audio source')
    parser.add_argument('--session', choices=['23','24'], default='24', type=str, help='The recording session to use')
    parser.add_argument('-o', '--output_dir', type=str, help='The output directory')
    parser.add_argument('--thresh', '-t', type=float, help='The threshold for VAD')
    parser.add_argument('--nfft', type=int, default=1024, help='The FFT size to use for STFT')
    parser.add_argument('--no_norm', action='store_true', help='Disable matching of output to channel 1')
    parser.add_argument('--clip_gain', type=float, help='Clip the maximum gain')
    parser.add_argument('--vad_guard', type=int, help='Value by which to extend VAD boundaries')
    parser.add_argument('--synth_mix', action='store_true', help='Works on artifical mix of signals and compute objective evaluation.')
    parser.add_argument('--save_sample', type=str, help='Save samples of input and output signals, argument is the directory')
    parser.add_argument('--plot', action='store_true', help='Display all the figures')
    parser.add_argument('--all', action='store_true', help='Process all the samples')
    args = parser.parse_args()

    if args.all:
        # store the results to save in JSON and later load in
        # pandas.DataFrame
        results = dict(
                columns=['Microphones', 'Algorithm', 'SIR_in', 'SDR', 'SIR'],
                data=[]
                )
        for mic in mic_choices.keys():
            for SIR in sir_choices:
                print('Running for mic={} SIR={} ...'.format(mic, SIR), end='')
                ret = process_experiment_max_sinr(SIR, mic, blinky_sig, args)
                for algo, metrics in ret.items():
                    results['data'].append([mic, algo, SIR, metrics['SDR'], metrics['SIR']])
                print('done.')
                for algo, metrics in ret.items():
                    print('{} SDR={:.2f} SIR={:.2f}'.format(algo, metrics['SDR'], metrics['SIR']))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        import datetime
        now = datetime.datetime.now()
        date_str = datetime.datetime.strftime(now, '%Y%m%d-%H%M%S')
        fn = '{}_results_experiment_sir.json'.format(date_str)
        filename = os.path.join(args.output_dir, fn)

        git_commit = get_git_hash()

        parameters = dict(
                git_commit=git_commit,
                nfft=args.nfft,
                vad_guard=args.vad_guard,
                clip_gain=args.clip_gain,
                thresh=args.thresh,
                no_norm=args.no_norm,
                synth_mix=args.synth_mix,
                )

        record = dict(
                date=date_str,
                parameters=parameters,
                results=results,
                )

        with open(filename, 'w') as f:
            json.dump(record, f, indent=2)

    else:
        try:
            SIR = args.SIR
            mic = args.mic
        except:
            raise ValueError('When the keyword --all is not used, SIR and mic are required arguments')

        ret = process_experiment_max_sinr(SIR, mic, blinky_sig, args)

        for algo, metrics in ret.items():
            print('{} SDR={:.2f} SIR={:.2f}'.format(algo, metrics['SDR'], metrics['SIR']))



'''
Make a webpage with all the samples
'''
import os, glob, argparse
import numpy as np
from scipy.io import wavfile

tgt_dtype = np.int16

def breakout_filename(filename):
    ''' Infer metadata from filename '''

    name = os.path.splitext(os.path.split(filename)[1])[0]

    if name.startswith('camera'):
        array, algorithm, _, SIR, __ = name.split('_')
        nchannels = 2
    else:
        array, nchannels, algorithm, _, SIR, __ = name.split('_')

    return {
            'array' : array,
            'nchannels' : int(nchannels),
            'algorithm' : algorithm,
            'SIR' : SIR,
            }


def reject_sample(metadata):
    ''' selects which files to reject based on metadata '''

    if metadata['array'] != 'pyramic':
        return True

    if metadata['algorithm'] == 'ch0':
        if not (metadata['array'] == 'pyramic' and metadata['nchannels'] == 2):
            return True

    print(metadata)

    return False

def audio_widget(filename):
    return '<audio controls="controls" type="audio/wav" src="{}"><a>play</a></audio>'.format(filename)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepares audio samples for release only')
    parser.add_argument('dir', type=str, help='The directory holding all the samples')
    parser.add_argument('out_dir', type=str, help='The directory where to output the prepared files')
    parser.add_argument('server_path', type=str, help='Paths to audio samples on the webserver')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    all_files = glob.glob(args.dir + '*.wav')

    all_metadata = dict()

    # first, move and convert all the samples
    for fn in all_files:
        metadata = breakout_filename(fn)
        bn = os.path.basename(fn)

        if reject_sample(metadata):
            continue

        fs, audio = wavfile.read(fn)

        if audio.dtype != np.int16:
            audio = (audio * (2 ** 15 - 1)).astype(np.int16)

        new_fn = os.path.join(args.out_dir, bn)
        wavfile.write(new_fn, fs, audio)

        SIR = metadata.pop('SIR')
        server_fn = os.path.join(args.server_path, bn)
        metadata['filename'] = server_fn

        if SIR not in all_metadata:
            all_metadata[SIR] = []

        all_metadata[SIR].append(metadata)

    # Second, create the html file
    with open(os.path.join(args.out_dir, 'index.html'), 'w') as f:
        f.write('''
<html>
    <head>
        <title>APSIPA 2018 Samples</title>
    </head>
    <body>
        <style type="text/css">
        audio {
            width: 50px;
        }
        </style>

        <table>
            <tr>
                <td>SIR</td>
                <td>Target</td>
                <td>Mix</td>
                <td colspan="2">AuxIVA</td>
                <td colspan="2">Max-SINR</td>
            </tr>

            <tr>
                <td></td>
                <td></td>
                <td></td>
                <td>2 ch</td>
                <td>4 ch</td>
                <td>2 ch</td>
                <td>4 ch</td>
                <td>24 ch</td>
                <td>48 ch</td>
            </tr>
''')


        SIR_points = list(all_metadata.keys())
        SIR_points.remove('NA')
        SIR_points = sorted(SIR_points, key=int)

        cols = [
                ['mix', 2],
                ['bss', 2], ['bss', 4],
                ['maxsinr', 2], ['maxsinr', 4], ['maxsinr', 24], ['maxsinr', 48],
                ]

        for SIR in SIR_points:
            f.write((' ' * 12) + '<tr>\n')
            f.write((' ' * 16) + '<td>' + SIR + '</td>\n')

            f_ref = all_metadata['NA'][0]['filename']
            f.write((' ' * 16) + '<td>' + audio_widget(f_ref) + '</td>\n')


            for col in cols:
                match = lambda x : x['algorithm'] == col[0] and x['nchannels'] == col[1]
                fn = list(filter(match, all_metadata[str(SIR)]))[0]['filename']
                f.write((' ' * 16) + '<td>' + audio_widget(fn) + '</td>\n')

            f.write((' ' * 12) + '</tr>\n')

        f.write('''
        </table>
    </body>
</html>''')




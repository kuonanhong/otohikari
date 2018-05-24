import json
import pyroomacoustics as pra
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def norm_pdf(p, x):
    return np.exp( -((x - p[0]) / p[1]) ** 2 ) / (np.sqrt(2. * np.pi) * p[1])

def error_pdf(p, x, y):

    return norm_pdf(p, x) - y

def error_pdf_jac(p, x, y):

    e = (x - p[0]) / p[1]

    J = np.zeros((x.shape[0], p.shape[0]))
    J[:,0] = norm_pdf(p, x) * e / p[1]
    J[:,1] = norm_pdf(p, x) * (e ** 2 - 1) / p[1]

    return J

def const_poly_fit(x, y, dim):
    x = np.array(x)

    A = x[:,None] ** np.arange(0, dim+1)

    c = np.linalg.lstsq(A, y)[0]
    fit_curve = np.dot(A, c)

    return c, fit_curve

if __name__ == '__main__':

    #cmu = pra.datasets.CMUArcticCorpus(basedir='/Volumes/datanet/CMU_ARCTIC', build=True, speaker=['ahw', 'lnh'])
    timit = pra.datasets.TimitCorpus(basedir='/Users/scheibler/PHD/Projects/Beamforming/timit/TIMIT')
    timit.build_corpus()
    chunk_size = 64
    chunks = []
    noise = []

    #for sample in cmu.samples:
    for sample in timit.sentence_corpus['TRAIN']:

        noise.append(np.var(sample.data[:200]))

        L = (sample.data.shape[0] // chunk_size) * chunk_size

        v = np.var(sample.data[:L].reshape((-1,chunk_size)), axis=1)

        chunks.append(v[v > 0.])

    chunks = np.concatenate(chunks)
    max_chunks = np.max(chunks)
    chunks /= max_chunks

    chunks_db = 10. * np.log10(chunks)

    noise = np.array(noise) / max_chunks

    thresh = -60
    
    y, x, _ = plt.hist(chunks_db[chunks_db > thresh], bins=10000, density=True, cumulative=True)
    x = (x[:-1] + x[1:]) / 2

    dim = 10
    c = np.polyfit(x, y, dim)

    A = x[:,None] ** np.arange(0, dim+1)
    y_fit = np.dot(A, c[::-1])

    plt.plot(x, y_fit, 'r')

    on_grid = np.polyval(c, np.arange(-60, 0))
    lut = [[l,u] for l,u in zip(on_grid[:-1], on_grid[1:])]

    with open('lut_audio2pwm.json', 'w') as f:
        info = {
                'base' : thresh,
                'lut' : lut,
                'coef' : c.tolist(),
                }
        json.dump(info, f, indent=1)

    # print the value for insertions in the C code

    print('float lut = {')
    val_list = on_grid.tolist()
    count = 0
    while len(val_list) > 1:
        v = val_list.pop(0)
        print('{},'.format(v), end='')
        count += 1
        if count % 10 == 0:
            print()
            print('  ', end='')
        else:
            print(' ', end='')

    print('{}'.format(val_list[0]))
    print('};')

    plt.show()



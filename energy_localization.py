'''
Energy based localization with sound-to-light conversion sensors
'''

import argparse
import numpy as np
from scipy.io import wavfile
from scipy import signal as sig
from scipy import linalg as la
from scipy.optimize import least_squares, basinhopping
import pyroomacoustics as pra
import matplotlib.pyplot as plt

from light_array import LightArray
from energy_localization_helpers import *
from pylocus.algorithms import procrustes
from pylocus.lateration import SRLS

parser = argparse.ArgumentParser()
parser.add_argument('--perfect_model', action='store_true', help='Use a perfect model for the data generation')
parser.add_argument('--true_init', action='store_true', help='Use groundtruth as initialization for the optimization')
args = parser.parse_args()

### Parameters ###
fs_sound = 16000
fs_light = 30
nfft = 512  # approximately 30 overlapping frames / second

r, target_audio = wavfile.read('../audio/DR1_FAKS0_SX133_SX313_SX223.wav')
assert r == fs_sound
target_audio = np.concatenate([np.zeros(fs_sound), target_audio.astype(np.float32)])
target_audio /= target_audio.max()
sigma_s = np.std(target_audio)
sound_time = np.arange(target_audio.shape[0]) / fs_sound

device_locations = np.c_[
        [ 1.5, 1.5 ],
        [ 2.5, 1.5 ],
        [ 3.5, 1.5 ],
        [ 4.0, 2.5 ],
        [ 3.5, 3.5 ],
        [ 2.5, 3.5 ],
        [ 1.5, 3.5 ],
        [ 1.0, 2.5 ],
        ]
n_devices = device_locations.shape[1]
device_gains_db = np.r_[ 0., 1.5, -2, -1.7, -0.5, 0.9, 2 , 0.1]
#device_gains_db = np.zeros(n_devices)
assert device_gains_db.shape[0] == n_devices

source_locations = np.c_[
        [ 1.35, 1.2 ],
        [ 2.60, 1.6 ],
        [ 3.40, 1.3 ],
        [ 4.20, 2.1 ],
        [ 3.35, 3.2 ],
        [ 2.20, 3.37 ],
        [ 1.60, 3.0 ],
        [ 0.90, 2.7 ],
        ]
n_speakers = source_locations.shape[1]
source_gains_db = np.r_[ -0.7, 1.1, 1.3, 0.2, -1.3, 0.2, 0.99, -0.1]
#source_gains_db = np.r_[ -0.7, 1.1, 1.3, 0.2, -1.3, 0.2, 0.99,]
assert source_gains_db.shape[0] == n_speakers

# for now, we assume 1 device / speaker
assert n_speakers <= n_devices

# Create the room
room = pra.ShoeBox([6,5], fs=16000, max_order=12, absorption=0.4, sigma2_awgn=1e-5)

# The active segments in seconds for each speakers
# For now, each speaker speaks for 0.8 seconds in turn
active_segments = [ ]
signals = np.zeros((n_devices, n_speakers * fs_sound))
for i,loc in enumerate(source_locations.T):

    lo = i + 0.1
    hi = i + 0.9
    active_segments.append([lo, hi])

    lo_i = int(lo * fs_sound)
    hi_i = int(hi * fs_sound)
    gain_lin = 10**(source_gains_db[i] / 10)
    signals[i,lo_i:hi_i] = np.random.randn(hi_i - lo_i) * gain_lin

    # add the source to the room
    room.add_source(loc, signal=signals[i])

# sound-to-light sensor
# we assume there is no propagation delay between speaker and sensor
devices = LightArray(device_locations, fs=fs_light)
room.add_microphone_array(devices)

# Simulate sound transport
room.simulate()

###############################
## ENERGY BASED LOCALIZATION ##
###############################
'''
Following:
ENERGY-BASED POSITION ESTIMATION OF MICROPHONES AND SPEAKERS FOR AD HOC MICROPHONE ARRAYS
Chen et al.
2007
'''
alpha_gt = 0.5

# Step 0 : A few groundtruth value to test
D2_gt = pra.distance(device_locations, source_locations)**2
A_gt = np.zeros((n_devices, n_speakers))
sigma_gt = np.ones((n_devices, n_speakers))
for i in range(n_devices):
    for j in range(n_speakers):
        A_gt[i,j] = device_gains_db[i] - alpha_gt * np.log(D2_gt[i,j]) + source_gains_db[j]
        if i == 0 and j == 0:
            print(A_gt[0,0])
A00 = A_gt[0,0]

def check_A(i,j):
    check_A.count_calls += 1
    print(check_A.count_calls, 'Matching? (',i,',',j,') ',A[i,j], device_gains_db[i] - alpha_gt * np.log(D2_gt[i,j]) + source_gains_db[j])
check_A.count_calls = 0

# Step 1 : Speaker segmentation
A = np.zeros((n_devices, n_speakers))
sigma = np.zeros((n_devices, n_speakers))
for i in range(n_devices):
    for j in range(n_speakers):
        lo, hi = [int(x * fs_light) for x in active_segments[j]]  # We have perfect segmentation for now
        A[i,j] = np.mean(devices.signals[i][lo:hi] / (20 / np.log(10)))  # energy is already in log domain
        sigma[i,j] = np.std(devices.signals[i][lo:hi] / (20 / np.log(10)))

if args.perfect_model:
    A = A_gt
    sigma = sigma_gt

# Step 2 : Initialization
var_shapes = [(n_devices), (n_speakers), (2, n_devices), (2, n_speakers), (1,)] 
packer = VarPacker(var_shapes)
x0 = packer.new_vector()
m0, s0, R0, X0, alpha0 = packer.unpack(x0)

alpha0[:] = 0.5

C = np.zeros((n_devices, n_speakers))  # log attenuation of speaker at microphone
D2 = np.zeros((n_devices, n_speakers))  # squared distance between speaker and microphone
for i in range(n_devices):
    for j in range(n_speakers):
        if i < n_speakers:
            C[i,j] = 0.5 * (A[i,j] + A[j,i] - A[i,i] - A[j,j])
        else:
            C[i,j] = A[i,j] - A[j,j]
        D2[i,j] = np.exp(-C[i,j] / alpha0[0])  # In practice, we'll need to find a way to fix the scale

# log gain of device
m0[0] = 0.  # i.e. m[0] = log(1)
for i in range(1,n_devices):
    m0[i] = A[i,0] - A[0,0] - C[i,0] + m0[0]

# energy of speaker
for j in range(n_speakers):
    s0[j] = np.mean(A[:,j] - m0[:] - C[:,j])

# Step 3 : MDS for intial estimate of microphone locations

# Fix microphone locations
R0[:,:] = device_locations
# Use SRLS to find intial guess of locations
for i in range(n_speakers):
    X0[:,i] = SRLS(R0.T, np.ones(n_devices), D2[:,i])

# We can test with initialization with true values
x_true = packer.new_vector()
m_gt, s_gt, R_gt, X_gt, alpha2_gt = packer.unpack(x_true)
m_gt[:] = device_gains_db[:]
s_gt[:] = source_gains_db[:]
R_gt[:,:] = device_locations[:,:]
X_gt[:,:] = source_locations[:,:]
alpha2_gt[:] = alpha_gt
if args.true_init:
    np.copyto(x0, x_true)

# Create a variable to loop over
x = packer.new_vector()
m, s, R, X, alpha = packer.unpack(x)
x[:] = x0

# Step 4 : Now we run the big non-linear optimization stuff
for i in range(20):

    # zero gradient for m and s
    S = A + alpha * np.log(pra.distance(R0, X)**2)
    m[:], s[:] = cdm_unfolding(1 / sigma, S, sum_matrix=True)

    # Non-linear least squares solver
    res_1 = least_squares(objective, x, jac=jacobian, 
            args=(A, sigma, packer),
            kwargs={'fix_mic' : True, 'fix_alpha' : i < 2},
            xtol=1e-15,
            ftol=1e-15, 
            max_nfev=1000,
            method='lm', 
            verbose=1)

    # use result as next initialization
    x[:] = res_1.x


print('RMSE SRLS : {} RMSE Non-Lin : {}'.format(
    np.sqrt(np.mean(np.linalg.norm(X0 - source_locations, axis=0)**2)),
    np.sqrt(np.mean(np.linalg.norm(X - source_locations, axis=0)**2)),
    ))

print('Objective value real {} vs found {}'.format(
    squared_loss(x_true, A, sigma, packer, objective, jacobian)[0],
    squared_loss(x, A, sigma, packer, objective, jacobian)[0],
    ))

plt.figure()
ax = pra.experimental.PointCloud(X=device_locations).plot()
pra.experimental.PointCloud(X=source_locations).plot(axes=ax)
pra.experimental.PointCloud(X=X0).plot(axes=ax)
pra.experimental.PointCloud(X=X).plot(axes=ax)
ax.axes.set_aspect('equal')
plt.legend(['device gt', 'source gt', 'sources SRLS', 'sources non-lin opt', '2nd round'])

plt.show()


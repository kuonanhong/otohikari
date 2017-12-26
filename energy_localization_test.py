'''
Energy based localization with sound-to-light conversion sensors
'''

import argparse
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

from light_array import LightArray
from energy_localization_helpers import *
from pylocus.lateration import SRLS
from rescaled_srls import rescaled_SRLS

parser = argparse.ArgumentParser()
parser.add_argument('--perfect_model', action='store_true', help='Use a perfect model for the data generation')
parser.add_argument('--true_init', action='store_true', help='Use groundtruth as initialization for the optimization')
parser.add_argument('--random_sources', action='store_true', help='Draw random source locations.')
args = parser.parse_args()

### Parameters ###
fs_sound = 16000
fs_light = 30
nfft = 512  # approximately 30 overlapping frames / second

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
assert device_gains_db.shape[0] == n_devices

source_locations = np.c_[
        [ 1.35, 1.2 ],
        [ 2.60, 1.6 ],
        [ 3.40, 1.3 ],
        [ 4.20, 2.1 ],
        [ 3.35, 3.2 ],
        [ 2.20, 3.37 ],
        [ 1.65, 2.9 ],
        [ 0.90, 2.7 ],
        ]
n_speakers = source_locations.shape[1]
source_gains_db = np.r_[ -0.7, 1.1, 1.3, 0.2, -1.3, 0.2, 0.99, -0.1]
assert source_gains_db.shape[0] == n_speakers

# for now, we assume 1 device / speaker
assert n_speakers <= n_devices

# Create the room
room_dim = np.array([6,5])
room = pra.ShoeBox(room_dim, fs=16000, max_order=12, absorption=0.4, sigma2_awgn=1e-5)

if args.random_sources:
    for i in range(n_speakers):
        success = False
        while not success:
            theta = np.random.rand() * 2 * np.pi
            p = np.array([np.cos(theta), np.sin(theta)])
            success = room.is_inside(p)
        source_locations[:,i] = device_locations[:,i] + p + np.random.randn(2) * 0.2

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

# Step 1 : Speaker segmentation
A = np.zeros((n_devices, n_speakers))
sigma = np.zeros((n_devices, n_speakers))
for i in range(n_devices):
    for j in range(n_speakers):
        lo, hi = [int(x * fs_light) for x in active_segments[j]]  # We have perfect segmentation for now
        #A[i,j] = np.mean(devices.signals[i][lo:hi] / (20 / np.log(10)))  # energy is already in log domain
        #sigma[i,j] = np.std(devices.signals[i][lo:hi] / (20 / np.log(10)))
        A[i,j] = np.mean(devices.signals[i][lo:hi])  # energy is already in log domain
        sigma[i,j] = np.std(devices.signals[i][lo:hi])

if args.perfect_model:
    A = A_gt
    sigma = sigma_gt

# run the algorithm
m, s, X, X0 = energy_localization(A, sigma, device_locations)

print('RMSE SRLS : {} RMSE Non-Lin : {}'.format(
    np.sqrt(np.mean(np.linalg.norm(X0 - source_locations, axis=0)**2)),
    np.sqrt(np.mean(np.linalg.norm(X - source_locations, axis=0)**2)),
    ))

plt.figure()
ax = pra.experimental.PointCloud(X=device_locations).plot()
pra.experimental.PointCloud(X=source_locations).plot(axes=ax)
pra.experimental.PointCloud(X=X0).plot(axes=ax)
pra.experimental.PointCloud(X=X).plot(axes=ax)
ax.axes.set_aspect('equal')
plt.legend(['device gt', 'source gt', 'sources SRLS', 'sources non-lin opt', '2nd round'])

plt.show()


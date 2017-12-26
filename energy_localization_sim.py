import os
import numpy as np

import run_parallel

# find the absolute path to this file
base_dir = os.path.abspath(os.path.split(__file__)[0])

def init(parameters):
    global base_dir
    parameters['base_dir'] = base_dir

def parallel_loop(args):
    global parameters

    # shorthand
    p = parameters

    # system imports
    import numpy
    from pyroomacoustics import ShoeBox
    import sys
    sys.path.append(p['base_dir'])

    # local imports
    from light_array import LightArray
    from energy_localization_helpers import energy_localization
    from rescaled_srls import rescaled_SRLS

    # locations of sources and receivers
    source_locations = args[0]
    source_gains_db = args[1]
    n_sources = source_locations.shape[1]

    device_locations = np.array(p['device_locations']).T
    device_gains_db = np.array(p['device_gains_db'])
    n_devices = devices.shape[1]

    # STEP 1 : Room simulation
    # Create the room
    room = ShoeBox(p['room_dim'], 
            fs=p['fs_sound'], 
            max_order=p['room_max_order'], 
            absorption=p['room_absorption'], 
            sigma2_awgn=p['sigma2_noise'])

    # The active segments in seconds for each speakers
    # For now, each speaker speaks for 0.8 seconds in turn
    active_segments = [ ]
    signals = numpy.zeros((n_devices, n_sources * fs_sound))
    for i,loc in enumerate(source_locations.T):

        lo = i + 0.1
        hi = i + 0.9
        active_segments.append([lo, hi])

        lo_i = int(lo * fs_sound)
        hi_i = int(hi * fs_sound)
        gain_lin = 10**(source_gains_db[i] / 10)
        signals[i,lo_i:hi_i] = numpy.random.randn(hi_i - lo_i) * gain_lin

        # add the source to the room
        room.add_source(loc, signal=signals[i])

    # sound-to-light sensor
    # we assume there is no propagation delay between speaker and sensor
    devices = LightArray(device_locations, fs=fs_light)
    room.add_microphone_array(devices)

    # Simulate sound transport
    room.simulate()

    # STEP 2 : Speaker segmentation
    A = numpy.zeros((n_devices, n_sources))
    sigma = numpy.zeros((n_devices, n_sources))
    for i in range(n_devices):
        for j in range(n_sources):
            lo, hi = [int(x * fs_light) for x in active_segments[j]]  # We have perfect segmentation for now
            A[i,j] = numpy.mean(devices.signals[i][lo:hi])  # energy is already in log domain
            sigma[i,j] = numpy.std(devices.signals[i][lo:hi])

    # STEP 3 : Source Localization
    m, s, X, X0 = energy_localization(A, sigma, device_locations,
            n_iter=p['energy_localization_n_iter'])

    # STEP 4 : Result processing
    results = dict(
            groundtruth_locations=source_locations.T.tolist(),
            reconstructed_locations=X.T.tolist(),
            )

    return results

def gen_args(parameters):

    import pyroomacoustics as pra

    room = pra.ShoeBox(parameters['room_dim'])
    device_locations = parameters['device_locations']
    n_dim = device_locations.shape[0]
    n_sources = parameters['n_sources']

    assert n_dim == 2, 'Simulation only supports 2D for now'

    gain_range = np.array(parameters['source_gain_range'])
    def map_range(x, I):
        return x * (I[1] - I[0]) + I[0]

    args = []

    for epoch in range(parameters['n_loops']):
        source_gains_db = map_range(np.random.rand(n_sources), gain_range)

        source_locations = np.zeros((n_dim,n_sources))
        for i in range(n_sources):
            success = False
            while not success:
                theta = np.random.rand() * 2 * np.pi
                p = np.array([np.cos(theta), np.sin(theta)])
                candidate = device_locations[:,i] + p + np.random.randn(2) * 0.2
                success = room.is_inside(candidate)
            source_locations[:,i] = candidate

        args.append([source_locations, source_gains_db])


    return args


if __name__ == '__main__':

    run_parallel.run(parallel_loop, gen_args, func_init=init, base_dir=base_dir, results_dir='data/', 
            description='Energy based localization evaluation simulation')

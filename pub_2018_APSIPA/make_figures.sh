#!/bin/bash

SAMPLES_DIR=pub_2018_APSIPA/samples
FIGURES_DIR=pub_2018_APSIPA/figures
DATA_DIR=pub_2018_APSIPA/data

## Beamforming experiment with target being the source (output channel) 1
python pub_2018_APSIPA/experiment_max_sinr.py ch1 --all \
  --vad_guard=3000 --nfft=2048 \
  --save_sample ${SAMPLES_DIR} \
  --output_dir ${DATA_DIR}

## Plot the result of the previous experiment
python pub_2018_APSIPA/figure_experiment_sir.py ${DATA_DIR}/20180601-004939_results_experiment_sir.json ${FIGURES_DIR}

## The speech power distribution figure
# python led_calibration/speech_distribution.py --save ${FIGURES_DIR} --no_lut

## The calibration figure before correction
# python led_calibration/led_calibration.py led_calibration/20180507_led_calibration_2_curves.json \
#  --plot --pwm 12 --color red --no_fit \
#  --save_plot ${FIGURES_DIR}

## The calibration figure after correction
# python led_calibration/led_calibration.py led_calibration/20180518_led_calibration_3_curves.json \
#  --plot --pwm 0 --color red --no_fit \
#  --save_plot ${FIGURES_DIR}

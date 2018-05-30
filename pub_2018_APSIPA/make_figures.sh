#!/bin/bash

SAMPLES_DIR=pub_2018_APSIPA/samples
FIGURES_DIR=pub_2018_APSIPA/figures
DATA_DIR=pub_2018_APSIPA/data

# Beamforming experiment with target being the source (output channel) 1
#
python pub_2018_APSIPA/experiment_max_sinr.py ch1 --all \
  --vad_guard=3000 --nfft=2048 \
  --save_sample ${SAMPLES_DIR} \
  --output_dir ${DATA_DIR}


#!/bin/bash

echo $0
CODE_DIR=$(dirname $0)
EXP_DIR=$1

# This extracts the blinky tracks from the video, as
# well as the audio source locations when the RC car is used
python ${DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v noise --track
python ${DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v speech --track
python ${DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_1
python ${DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_2
python ${DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_3
python ${DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_4
python ${DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_5

# Prepare the data extracted from the videos to be fed to the DNN
python ${DIR}/data_preparation.py ${EXP_DIR}/protocol.json -n 1

# Train the DNN
python ${DIR}/../ml_localization/train.py ${DIR}/dnn/config/resnet_dropout.json

# Evaluate on the test set
python ${DIR}/test.py ${EXP_DIR}/protocol.json ${DIR}/dnn/config/resnet_dropout.json

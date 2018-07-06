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

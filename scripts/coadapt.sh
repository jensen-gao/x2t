#!/bin/bash

# X2T on Default Interface Data
for i in {0..11}
do
  python train.py -u og -s x2t_on_default_data/$i -l experiments/gaze_study/x2t/$i/offline_model.zip -gp experiments/gaze_study/default/$i/
done

# Default Interface on X2T Data
for i in {0..11}
do
  python train.py -u og -s default_on_x2t_data/$i -m b -gp experiments/gaze_study/x2t/$i/
done

#!/bin/bash

for i in {0..11}
do
  for j in {0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5}
  do
    python train.py -u og -s rew_noise/$j/$i -l experiments/gaze_study/x2t/$i/offline_model.zip -er $j -gp experiments/gaze_study/x2t/$i/
  done
done

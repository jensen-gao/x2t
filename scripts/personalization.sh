#!/bin/bash

for i in {0..3}
do
  for j in {0..3}
  do
    python train.py -u su -s personalization/$j/$i -be -lm -m m -ui $i -sd $((i+60)) -ds $((i+60)) -l experiments/sim_uji/x2t/$j/final_model.zip -dl
  done
done

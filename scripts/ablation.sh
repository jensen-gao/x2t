#!/bin/bash

# Default Interface
for i in {0..59}
do
  python train.py -u su -s default/$i -be -lm -m b -ui $i -sd $i -ds $i
done

# Offline Pretraining
for i in {0..59}
do
  python train.py -u su -oo -s pretrain/$i -sd $i -dp experiments/sim_uji/default/$i/data.hdf5
done

# Full X2T
for i in {0..59}
do
  python train.py -u su -s x2t/$i -be -lm -m m -ui $i -sd $((i+60)) -ds $((i+60)) -l experiments/sim_uji/pretrain/$i/offline_model.zip
done

# X2T without Offline Pretraining
for i in {0..59}
do
    python train.py -u su -s x2t_no_pretrain/$i -be -lm -m m -ui $i -sd $((i+60)) -ds $((i+60))
done

# X2T without Online Learning
for i in {0..59}
do
    python train.py -u su -s x2t_no_online/$i -be -lm -m m -ui $i -sd $((i+60)) -ds $((i+60)) -l experiments/sim_uji/pretrain/$i/offline_model.zip -dl
done

# X2T without Prior Policy
for i in {0..59}
do
    python train.py -u su -s x2t_no_prior/$i -be -m l -ui $i -sd $((i+60)) -ds $((i+60)) -l experiments/sim_uji/pretrain/$i/offline_model.zip
done

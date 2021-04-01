# Modified from the Transformer-XL code found at https://github.com/kimiyoung/transformer-xl

#!/bin/bash

URL=http://curtis.ml.cmu.edu/datasets/pretrained_xl

DATA_ROOT=transformer_xl/

function download () {
  fileurl=${1}
  filename=${fileurl##*/}
  if [ ! -f ${filename} ]; then
    echo ">>> Download '${filename}' from '${fileurl}'."
    wget --quiet ${fileurl}
  else
    echo "*** File '${filename}' exists. Skip."
  fi
}

cd $DATA_ROOT
mkdir -p pretrained_xl && cd pretrained_xl

# lm1b
mkdir -p tf_lm1b && cd tf_lm1b

mkdir -p data && cd data
download ${URL}/tf_lm1b/data/cache.pkl
download ${URL}/tf_lm1b/data/corpus-info.json
cd ..

mkdir -p model && cd model
download ${URL}/tf_lm1b/model/checkpoint
download ${URL}/tf_lm1b/model/model.ckpt-1191000.data-00000-of-00001
download ${URL}/tf_lm1b/model/model.ckpt-1191000.index
download ${URL}/tf_lm1b/model/model.ckpt-1191000.meta
cd ..

cd ..

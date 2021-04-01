import os
import sys
import pickle
from transformer_xl.tf import data_utils
from transformer_xl.tf import vocabulary


# Script for fixing the cache file used by the One Billion Words language model to work with the
# project's directory structure.

sys.modules['data_utils'] = data_utils
sys.modules['vocabulary'] = vocabulary

data_dir = 'transformer_xl/pretrained_xl/tf_lm1b/data'
fn = os.path.join(data_dir, "cache.pkl")
with open(fn, "rb") as fp:
    corpus = pickle.load(fp, encoding='latin1')
del sys.modules['data_utils']
del sys.modules['vocabulary']
with open(fn, "wb") as fp:
    pickle.dump(corpus, fp, protocol=2)

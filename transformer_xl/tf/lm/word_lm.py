import numpy as np
from transformer_xl.tf.lm.lm import LM


class WordLM(LM):
    """
    Word level language model using Transformer-XL trained on the One Billion Words Benchmark.
    """
    def __init__(self, data_dir='transformer_xl/pretrained_xl/tf_lm1b/data',
                 corpus_info_path='transformer_xl/pretrained_xl/tf_lm1b/data/corpus-info.json',
                 eval_ckpt_path='transformer_xl/pretrained_xl/tf_lm1b/model/model.ckpt-1191000',
                 model_dir='transformer_xl/pretrained_xl/tf_lm1b/model/',
                 div_val=4, untie_r=True, proj_share_all_but_first=False, proj_same_dim=False, n_layer=24, d_model=1280,
                 d_embed=1280, n_head=16, d_head=80, d_inner=8192, mem_len=128, clamp_len=-1,
                 same_length=True, init='normal', init_std=0.02, proj_init_std=0.01, init_range=0.1, dataset='lm1b',
                 vocab_size=60000, vocab=None):
        super(WordLM, self).__init__(data_dir, corpus_info_path, eval_ckpt_path, model_dir, div_val, untie_r,
                                     proj_share_all_but_first, proj_same_dim, n_layer, d_model, d_embed, n_head,
                                     d_head, d_inner, mem_len, clamp_len, same_length, init, init_std, proj_init_std,
                                     init_range, dataset, vocab_size, vocab)

    def get_logits(self, text):
        numerized = self.tokenize_and_numerize(text)[None]
        feed_dict = self.initial_feed_dict.copy()
        feed_dict[self.input] = numerized
        logits = self.sess.run(self.logits, feed_dict=feed_dict)[0]
        return logits

    def tokenize_and_numerize(self, text):
        if text:
            tokens = self.lm_vocab.tokenize(text)
            return self.lm_vocab.convert_to_nparray(tokens)
        else:
            return np.full(1, 1)

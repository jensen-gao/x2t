# Adapted from the Transformer-XL code found at https://github.com/kimiyoung/transformer-xl

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from transformer_xl.tf.eval_model import eval_transformer
from transformer_xl.tf.data_utils import get_lm_corpus, get_corpus_info
from transformer_xl.tf.gpu_utils import assign_to_gpu
import numpy as np
from abc import ABC


class LM(ABC):
    """
    Class for using pretrained Transformer-XL language models

    :param vocab_size: (int) Limit on vocab size. Truncates the vocabulary used by the pretrained model
        to its most common words.
    :param vocab: (List(str)) Limits the vocabulary used by the language model to only use these words. Overrides
        vocab_size.
    """
    def __init__(self, data_dir, corpus_info_path, eval_ckpt_path, model_dir, div_val, untie_r,
                 proj_share_all_but_first, proj_same_dim, n_layer, d_model, d_embed, n_head, d_head, d_inner, mem_len,
                 clamp_len, same_length, init, init_std, proj_init_std, init_range, dataset, vocab_size, vocab):

        self.div_val = div_val
        self.untie_r = untie_r
        self.proj_share_all_but_first = proj_share_all_but_first
        self.proj_same_dim = proj_same_dim
        self.n_layer = n_layer
        self.d_model = d_model
        self.d_embed = d_embed
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.mem_len = mem_len
        self.clamp_len = clamp_len
        self.same_length = same_length
        self.init = init
        self.init_std = init_std
        self.proj_init_std = proj_init_std
        self.init_range = init_range
        self.vocab_size = vocab_size

        corpus = get_lm_corpus(data_dir, dataset, self.vocab_size)
        self.lm_vocab = corpus.vocab
        if vocab is None:
            self.vocab_idx = None
        else:
            self.vocab_idx = [self.lm_vocab.get_idx(word) for word in vocab]

        corpus_info = get_corpus_info(corpus_info_path)
        n_token = corpus_info["vocab_size"]
        cutoffs = corpus_info["cutoffs"][1:-1]
        ps_device = "/gpu:0"

        # Create computational graph
        g = tf.Graph()
        with g.as_default():
            with tf.device(assign_to_gpu(0, ps_device)), tf.variable_scope(tf.get_variable_scope(),
                                                                           reuse=tf.AUTO_REUSE):
                self.input = tf.placeholder(tf.int64, [1, None])
                self.mems = [tf.placeholder(tf.float32, [mem_len, 1, d_model]) for _ in range(n_layer)]

                self.hidden, self.logits, self.new_mems = self.single_core_graph(
                    n_token=n_token,
                    cutoffs=cutoffs,
                    inp=self.input,
                    mems=self.mems)

            # Evaluation loop
            self.initial_mems_np = [np.zeros([self.mem_len, 1, self.d_model], dtype=np.float32)
                                    for _ in range(self.n_layer)]
            self.initial_feed_dict = {m: m_np for m, m_np in zip(self.mems, self.initial_mems_np)}

            saver = tf.train.Saver()

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=g)
            self.sess.run(tf.global_variables_initializer())

            if eval_ckpt_path is None:
                eval_ckpt_path = tf.train.latest_checkpoint(model_dir)
            else:
                eval_ckpt_path = eval_ckpt_path
            saver.restore(self.sess, eval_ckpt_path)

    # Tokenizes and numerizes a sequence of text, that does not consist of punctuation or multiple sentences
    def tokenize_and_numerize(self, text):
        pass

    def get_model_fn(self, n_token, cutoffs):
        def model_fn(inp, mems):
            inp = tf.transpose(inp, [1, 0])

            if self.init == "uniform":
                initializer = tf.initializers.random_uniform(
                    minval=-self.init_range,
                    maxval=self.init_range,
                    seed=None)
            elif self.init == "normal":
                initializer = tf.initializers.random_normal(
                    stddev=self.init_std,
                    seed=None)
                proj_initializer = tf.initializers.random_normal(
                    stddev=self.proj_init_std,
                    seed=None)

            tie_projs = [False for _ in range(len(cutoffs) + 1)]
            if self.proj_share_all_but_first:
                for i in range(1, len(tie_projs)):
                    tie_projs[i] = True

            hidden, logits, new_mems = eval_transformer(
                dec_inp=inp,
                mems=mems,
                n_token=n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_inner,
                initializer=initializer,
                proj_initializer=proj_initializer,
                mem_len=self.mem_len,
                cutoffs=cutoffs,
                div_val=self.div_val,
                tie_projs=tie_projs,
                input_perms=None,
                same_length=self.same_length,
                clamp_len=self.clamp_len,
                use_tpu=False,
                untie_r=self.untie_r,
                proj_same_dim=self.proj_same_dim,
                vocab_size=self.vocab_size,
                vocab_idx=self.vocab_idx)

            return hidden, logits, new_mems

        return model_fn

    def single_core_graph(self, n_token, cutoffs, inp, mems):
        model_fn = self.get_model_fn(
            n_token=n_token,
            cutoffs=cutoffs)

        model_ret = model_fn(
            inp=inp,
            mems=mems)

        return model_ret

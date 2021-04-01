# Adapted from the Transformer-XL code found at https://github.com/kimiyoung/transformer-xl

from transformer_xl.tf.model import *


def mask_adaptive_logits(hidden, n_token, d_embed, d_proj, cutoffs,
                         params, tie_projs, proj_initializer=None,
                         div_val=1, scope='adaptive_softmax',
                         proj_same_dim=True, vocab_size=None, vocab_idx=None, **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if proj is not None:
            y = tf.einsum('bd,ed->be', y, proj)
        return tf.einsum('bd,nd->bn', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.variable_scope(scope):
        if len(cutoffs) == 0:
            softmax_b = tf.get_variable('bias', [n_token],
                                        initializer=tf.zeros_initializer())
            logits = _logit(hidden, params_W, softmax_b, params_projs)
        else:
            cutoff_ends = [0] + cutoffs + [n_token]
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    if vocab_size and l_idx >= vocab_size:
                        break
                    cur_d_embed = d_embed // (div_val ** i)

                    if div_val == 1:
                        cur_W = params_W[l_idx: r_idx]
                    else:
                        cur_W = params_W[i]
                    cur_b = tf.get_variable('b', [r_idx - l_idx],
                                            initializer=tf.zeros_initializer())
                    if tie_projs[i]:
                        if div_val == 1:
                            cur_proj = params_projs
                        else:
                            cur_proj = params_projs[i]
                    else:
                        if (div_val == 1 or not proj_same_dim) and d_proj == cur_d_embed:
                            cur_proj = None
                        else:
                            cur_proj = tf.get_variable('proj', [cur_d_embed, d_proj],
                                                       initializer=proj_initializer)
                    if i == 0:
                        cluster_W = tf.get_variable('cluster_W', [len(cutoffs), d_embed],
                                                    initializer=tf.zeros_initializer())
                        cluster_b = tf.get_variable('cluster_b', [len(cutoffs)],
                                                    initializer=tf.zeros_initializer())
                        cur_W = tf.concat([cur_W, cluster_W], 0)
                        cur_b = tf.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(hidden, cur_W, cur_b, cur_proj)
                        head_logprob = tf.nn.log_softmax(head_logit)
                        logits = head_logprob[:, :cutoff_ends[1]]
                    else:
                        tail_logit = _logit(hidden, cur_W, cur_b, cur_proj)
                        tail_logprob = tf.nn.log_softmax(tail_logit)
                        cluster_logprob = head_logprob[:, cutoff_ends[1] + i - 1]
                        logits = tf.concat([logits, cluster_logprob + tail_logprob], axis=1)
    if vocab_idx:
        logits = tf.gather(logits, vocab_idx, axis=1)  # Overrides vocab_size
    elif vocab_size:
        logits = logits[:, :vocab_size]
    return logits


def eval_transformer(dec_inp, mems, n_token, n_layer, d_model, d_embed,
                     n_head, d_head, d_inner,
                     initializer, proj_initializer=None,
                     mem_len=None, cutoffs=[], div_val=1, tie_projs=[],
                     same_length=False, clamp_len=-1, use_tpu=True,
                     input_perms=None, untie_r=False, proj_same_dim=True,
                     vocab_size=None, vocab_idx=None, scope='transformer'):
    """
    cutoffs: a list of python int. Cutoffs for adaptive softmax.
    tie_projs: a list of python bools. Whether to tie the projections.
    use_tpu: if True, use one_hot in embedding lookup and bin-based implementation
          of adaptive softmax.
    perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
          Only used in the adaptive setting.
    vocab_size: limit on vocab size, truncates the vocabulary to the most common words.
    vocab_idx: filters vocabulary to only these indices. Overrides vocab_size.
    """
    new_mems = []
    with tf.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                       initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                       initializer=initializer)
        else:
            r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                       initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                       initializer=initializer)

        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen

        if proj_initializer is None:
            proj_initializer = initializer
        lookup_fn = (mul_adaptive_embedding_lookup if use_tpu else
                     mask_adaptive_embedding_lookup)
        embeddings, shared_params = lookup_fn(
            x=dec_inp,
            n_token=n_token,
            d_embed=d_embed,
            d_proj=d_model,
            cutoffs=cutoffs,
            initializer=initializer,
            proj_initializer=proj_initializer,
            div_val=div_val,
            perms=input_perms,
            proj_same_dim=proj_same_dim)

        attn_mask = create_mask(qlen, mlen, same_length)

        pos_seq = tf.range(klen - 1, -1, -1.0)
        if clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, clamp_len)
        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        output = embeddings

        if mems is None:
            mems = [None] * n_layer

        for i in range(n_layer):
            # cache new mems
            new_mems.append(cache_mem(output, mems[i], mem_len))

            with tf.variable_scope('layer_{}'.format(i)):
                output = rel_multihead_attn(
                    w=output,
                    r=pos_emb,
                    r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                    r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                    attn_mask=attn_mask,
                    mems=mems[i],
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=0,
                    dropatt=0,
                    is_training=False,
                    kernel_initializer=initializer)
                output = positionwise_FF(
                    inp=output,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=0,
                    kernel_initializer=initializer,
                    is_training=False)

        hidden = output[-1]
        logits = mask_adaptive_logits(hidden=hidden,
                                      n_token=n_token,
                                      d_embed=d_embed,
                                      d_proj=d_model,
                                      cutoffs=cutoffs,
                                      params=shared_params,
                                      tie_projs=tie_projs,
                                      proj_initializer=proj_initializer,
                                      div_val=div_val,
                                      proj_same_dim=proj_same_dim,
                                      vocab_size=vocab_size,
                                      vocab_idx=vocab_idx)

        return hidden, logits, new_mems

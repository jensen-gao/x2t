import tensorflow as tf
from gym.spaces import MultiDiscrete

from stable_baselines.common import tf_util

# Code adapted from the Stable Baselines 2 implementation of DQN


def build_act(policy):
    """
    Creates the act function:

    :param policy: (DQNPolicy) the policy
    """
    _act = tf_util.function(inputs=[policy._obs_ph, policy.targets_ph,
                                    policy.train_ph],
                            outputs=[policy.q_values])

    def act(obs_seq, targets):
        return _act(obs_seq, targets, False)

    return act


def build_train(q_func, ob_space, ac_space, optimizer, sess, grad_norm_clipping=None,
                scope="deepq", reuse=None, full_tensorboard_log=False):
    """
    Creates the train function:

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) whether or not to reuse the graph variables
    :param optimizer: (tf.train.Optimizer) optimizer to use for the Q-learning objective.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param grad_norm_clipping: (float) clip gradient norms to this value. If None no clipping is performed.
    :param scope: (str or VariableScope) optional scope for variable_scope.
    :param reuse: (bool) whether or not the variables should be reused. To be able to reuse the scope must be given.
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :return: (tuple)

        act
        train: (function (Any, numpy float, numpy float, Any, numpy bool, numpy float): numpy float)
            optimize the error in Bellman's equation. See the top of the file for details.
        step_model: (DQNPolicy) Policy for evaluation
    """
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n

    with tf.variable_scope(scope, reuse=reuse):
        policy = q_func(sess, ob_space, ac_space, 1, 1, None)
        act = build_act(policy)

        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")

    with tf.variable_scope("loss", reuse=reuse):
        # set up placeholders
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        act_mask = tf.one_hot(act_t_ph, n_actions)
        labels = tf.nn.relu(tf.math.sign(rew_t_ph))
        dist = tf.nn.softmax(policy.q_values)
        pred = tf.reduce_sum(dist * act_mask, axis=1)
        loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=pred))

        tf.summary.scalar("loss", loss)

        # compute optimization op (potentially with gradient clipping)
        gradients = optimizer.compute_gradients(loss, var_list=q_func_vars)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))

        if full_tensorboard_log:
            tf.summary.histogram('rewards', rew_t_ph)

    optimize_expr = optimizer.apply_gradients(gradients)

    summary = tf.summary.merge_all()

    # Create callable functions
    _train = tf_util.function(
        inputs=[
            policy._obs_ph,
            act_t_ph,
            rew_t_ph,
            policy.targets_ph,
            policy.train_ph
        ],
        outputs=summary,
        updates=[optimize_expr]
    )

    def train(obses, actions, rewards, targets, **kwargs):
        return _train(obses, actions, rewards, targets, True, **kwargs)

    return act, train, policy

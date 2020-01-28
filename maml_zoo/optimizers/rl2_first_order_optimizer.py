from maml_zoo.logger import logger
from maml_zoo.optimizers.base import Optimizer
import numpy as np
import tensorflow as tf

class RL2FirstOrderOptimizer(Optimizer):
    """
    Optimizer for first order methods (SGD, Adam)

    Args:
        tf_optimizer_cls (tf.train.optimizer): desired tensorflow optimzier for training
        tf_optimizer_args (dict or None): arguments for the optimizer
        learning_rate (float): learning rate
        max_epochs: number of maximum epochs for training
        tolerance (float): tolerance for early stopping. If the loss fucntion decreases less than the specified tolerance
        after an epoch, then the training stops.
        num_minibatches (int): number of mini-batches for performing the gradient step. The mini-batch size is
        batch size//num_minibatches.
        verbose (bool): Whether to log or not the optimization process

    """

    def __init__(
            self,
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-6,
            num_minibatches=1,
            backprop_steps=32,
            verbose=False
    ):
        self._target = None
        if tf_optimizer_args is None:
            tf_optimizer_args = dict()
        tf_optimizer_args['learning_rate'] = learning_rate

        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._num_minibatches = num_minibatches  # Unused
        self._verbose = verbose
        self._all_inputs = None
        self._train_op = None
        self._policy_loss = None
        self._baseline_loss = None
        self._next_hidden_var = None
        self._hidden_ph = None
        self._input_ph_dict = None
        self._backprop_steps = backprop_steps

    def build_graph(self, policy_loss, baseline_loss, target, input_ph_dict, hidden_ph, next_hidden_var):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
        """
        assert isinstance(policy_loss, tf.Tensor)
        assert isinstance(baseline_loss, tf.Tensor)
        assert hasattr(target, 'get_params')
        assert isinstance(input_ph_dict, dict)

        self._target = target
        self._input_ph_dict = input_ph_dict
        self._policy_loss = policy_loss
        self._baseline_loss = baseline_loss
        self._hidden_ph = hidden_ph
        self._next_hidden_var = next_hidden_var
        params = list(target.get_params().values())
        self._gradients_var = tf.gradients(policy_loss + baseline_loss, params)
        self._gradients_ph = [tf.placeholder(shape=param.shape, dtype=tf.float32) for param in params]
        applied_gradients = zip(self._gradients_ph, params)
        self._train_op = self._tf_optimizer.apply_gradients(applied_gradients)

    def loss(self, input_val_dict):
        """
        Computes the value of the loss for given inputs

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float): value of the loss

        """
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        batch_size, seq_len, *_ = list(input_val_dict.values())[0].shape
        hidden_batch = self._target.get_zero_state(batch_size)
        feed_dict[self._hidden_ph] = hidden_batch
        policy_loss, baseline_loss = sess.run([self._policy_loss, self._baseline_loss], feed_dict=feed_dict)
        return policy_loss, baseline_loss

    def optimize(self, input_val_dict):
        """
        Carries out the optimization step

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float) loss before optimization

        """

        sess = tf.get_default_session()
        batch_size, seq_len, *_ = list(input_val_dict.values())[0].shape

        policy_loss_before_opt = None
        baseline_loss_before_opt = None
        for epoch in range(self._max_epochs):
            hidden_batch = self._target.get_zero_state(batch_size)
            if self._verbose:
                logger.log("Epoch %d" % epoch)
            # run train op
            policy_loss = []
            baseline_loss = []
            all_grads = []

            for i in range(0, seq_len, self._backprop_steps):
                n_i = i + self._backprop_steps
                # contains placeholders of things we need for training like advantage, baseline_targets
                feed_dict = dict([(self._input_ph_dict[key], input_val_dict[key][:, i:n_i]) for key in
                                  self._input_ph_dict.keys()])
                feed_dict[self._hidden_ph] = hidden_batch
                policy_batch_loss, baseline_batch_loss, grads, hidden_batch = sess.run([self._policy_loss, self._baseline_loss, self._gradients_var, self._next_hidden_var],
                                                            feed_dict=feed_dict)
                policy_loss.append(policy_batch_loss)
                baseline_loss.append(baseline_batch_loss)
                all_grads.append(grads)

            grads = [np.mean(grad, axis=0) for grad in zip(*all_grads)]
            feed_dict = dict(zip(self._gradients_ph, grads))
            _ = sess.run(self._train_op, feed_dict=feed_dict)

            if not policy_loss_before_opt: policy_loss_before_opt = np.mean(policy_loss)
            if not baseline_loss_before_opt: baseline_loss_before_opt = np.mean(baseline_loss)

            # if self._verbose:
            #     logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            #
            # if abs(last_loss - new_loss) < self._tolerance:
            #     break
            # last_loss = new_loss
        return policy_loss_before_opt, baseline_loss_before_opt


class RL2PPOOptimizer(RL2FirstOrderOptimizer):
    """
    Adds inner and outer kl terms to first order optimizer  #TODO: (Do we really need this?)

    """
    def __init__(self, *args, **kwargs):
        # Todo: reimplement minibatches
        super(RL2PPOOptimizer, self).__init__(*args, **kwargs)
        self._inner_kl = None
        self._outer_kl = None

    def build_graph(self, loss, target, input_ph_dict, inner_kl=None, outer_kl=None):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf.Tensor) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
            inner_kl (list): list with the inner kl loss for each task
            outer_kl (list): list with the outer kl loss for each task
        """
        super(RL2PPOOptimizer, self).build_graph(loss, target, input_ph_dict)
        assert inner_kl is not None

        self._inner_kl = inner_kl
        self._outer_kl = outer_kl

    def compute_stats(self, input_val_dict):
        """
        Computes the value the loss, the outer KL and the inner KL-divergence between the current policy and the
        provided dist_info_data

        Args:
           inputs (list): inputs needed to compute the inner KL
           extra_inputs (list): additional inputs needed to compute the inner KL

        Returns:
           (float): value of the loss
           (ndarray): inner kls - numpy array of shape (num_inner_grad_steps,)
           (float): outer_kl
        """
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss, inner_kl, outer_kl = sess.run([self._loss, self._inner_kl, self._outer_kl], feed_dict=feed_dict)
        return loss, inner_kl, outer_kl

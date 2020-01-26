import sys
import tensorflow as tf
from maml_zoo.utils.utils import get_original_tf_name, get_last_scope


def create_mlp(name,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               reuse=False
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    with tf.variable_scope(name):
        x = input_var

        for idx, hidden_size in enumerate(hidden_sizes):
            x = tf.layers.dense(x,
                                hidden_size,
                                name='hidden_%d' % idx,
                                activation=hidden_nonlinearity,
                                kernel_initializer=w_init,
                                bias_initializer=b_init,
                                reuse=reuse,
                                )

        output_var = tf.layers.dense(x,
                                     output_dim,
                                     name='output',
                                     activation=output_nonlinearity,
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     reuse=reuse,
                                     )

    return input_var, output_var


def forward_mlp(output_dim,
                hidden_sizes,
                hidden_nonlinearity,
                output_nonlinearity,
                input_var,
                mlp_params,
                ):
    """
    Creates the forward pass of an mlp given the input vars and the mlp params. Assumes that the params are passed in
    order i.e. [hidden_0/kernel, hidden_0/bias, hidden_1/kernel, hidden_1/bias, ..., output/kernel, output/bias]
    Args:
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        mlp_params (OrderedDict): OrderedDict of the params of the neural network.

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """
    x = input_var
    idx = 0
    bias_added = False
    sizes = tuple(hidden_sizes) + (output_dim,)

    if output_nonlinearity is None:
        output_nonlinearity = tf.identity

    for name, param in mlp_params.items():
        assert str(idx) in name or (idx == len(hidden_sizes) and "output" in name)

        if "kernel" in name:
            assert param.shape == (x.shape[-1], sizes[idx])
            x = tf.matmul(x, param)
        elif "bias" in name:
            assert param.shape == (sizes[idx],)
            x = tf.add(x, param)
            bias_added = True
        else:
            raise NameError

        if bias_added:
            if "hidden" in name:
                x = hidden_nonlinearity(x)
            elif "output" in name:
                x = output_nonlinearity(x)
            else:
                raise NameError
            idx += 1
            bias_added = False
    output_var = x
    return input_var, output_var # Todo why return input_var?


def create_rnn(name,
               cell_type,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               state_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               cnn_args=None,
               ):
    """
    Creates an RNN with optional CNN encoder on the front
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the recurrent modules
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, 64, 64, 3)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network
        cnn_args (dict): if not None, build cnn encoder
         - image_shape: H x W x C
         - base_depth: number of channels in the input data
         - double_camera: whether we take two images or one in the input

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None
    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')

    if state_var is None:
        create_hidden = True
    else:
        create_hidden = False

    build_cnn = (cnn_args is not None)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if build_cnn:
            # make the cnn
            conv_layers = []
            base_depth = cnn_args['base_depth']
            filter_mults = [1, 2, 4, 8]
            kernel_sizes = [5, 3, 3, 3]
            if not cnn_args['double_camera']:
                for f, k in zip(filter_mults, kernel_sizes):
                    conv_layers.append(tf.keras.layers.Conv2D(f * base_depth, k, 2, padding="SAME", activation=tf.nn.leaky_relu)) # number of output filters, kernel size, stride
                conv_layers.append(tf.keras.layers.Conv2D(8 * base_depth, 4, padding="VALID", activation=tf.nn.leaky_relu))
            '''
            else:
                self.conv1 = conv(base_depth, (10, 5), 2)  # conv: filters, kernel_size, stride
                self.conv2 = conv(2 * base_depth, (6, 3), 2)
                self.conv3 = conv(4 * base_depth, (6, 3), 2)
                self.conv4 = conv(8 * base_depth, (6, 3), 2)
                self.conv5 = conv(8 * base_depth, (8, 4), padding="VALID")
            '''

        # make the rnn
        cell = []
        if state_var is None:
            state_var = []

        for idx, hidden_size in enumerate(hidden_sizes):
            if cell_type == 'lstm':
                cell.append(tf.nn.rnn_cell.LSTMCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    c = tf.placeholder(tf.float32, (None, hidden_size), name='cell_state_%d' % idx)
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(tf.contrib.rnn.LSTMStateTuple(c, h))
            elif cell_type == 'gru':
                cell.append(tf.nn.rnn_cell.GRUCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(h)
            elif cell_type == 'rnn':
                cell.append(tf.nn.rnn_cell.RNNCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(h)
            else:
                raise NotImplementedError

        ###############################

        if build_cnn:
            image_shape = 64*64*3 # TODO
            output_feat_dim = 256
            print('ORIG input var', input_var.shape)
            # reshape the image part of the input vector into a 4-D tensor
            cnn_input_var = input_var[..., :image_shape]
            cnn_input_var = tf.reshape(cnn_input_var, (-1, 64, 64, 3)) # batch x h x w x c
            rest_input_var = input_var[..., image_shape:]
            print('CNN input var', cnn_input_var.shape)
            # pass the input var through the conv layers
            for layer in conv_layers:
                cnn_input_var = layer(cnn_input_var)

            # flatten resulting conv features
            flat_shape = (tf.shape(rest_input_var)[0], tf.shape(rest_input_var)[1], output_feat_dim)
            cnn_input_var = tf.reshape(cnn_input_var, flat_shape)  # batch x feature
            print('rest input var', rest_input_var.shape)
            print('CNN output', cnn_input_var.shape)

            # concat the rest of the ob with the image feature
            rnn_input_var = tf.concat((cnn_input_var, rest_input_var), axis=-1)
            print('post-cnn input', rnn_input_var.shape)

        else:
            rnn_input_var = input_var
        print('RNN input var', rnn_input_var.shape)

        # pass next through the rnn cells
        if len(hidden_sizes) > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            if create_hidden:
                state_var = tuple(state_var)
        else:
            cell = cell[0]
            if create_hidden:
                state_var = state_var[0]

        outputs, next_state_var = tf.nn.dynamic_rnn(cell,
                                        rnn_input_var,
                                        initial_state=state_var,
                                        time_major=False,
                                        )

        # pass through the final fully connected output layer
        output_var = tf.layers.dense(outputs,
                                     output_dim,
                                     name='output',
                                     activation=output_nonlinearity,
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     )

    return input_var, state_var, output_var, next_state_var, cell


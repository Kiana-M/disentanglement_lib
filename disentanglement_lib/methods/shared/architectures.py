# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of commonly used architectures and reconstruction losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf
import gin.tf
#for CapsuleNet:
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils

#requirements for caps encoder:

# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
    
class Mask(Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = inputs * K.expand_dims(mask, -1)
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config

class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )

    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs, 
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


#end of requirements for capsnet


@gin.configurable("encoder", whitelist=["num_latent", "encoder_fn"])
def make_gaussian_encoder(input_tensor,
                          is_training=True,
                          num_latent=gin.REQUIRED,
                          encoder_fn=gin.REQUIRED):
  """Gin wrapper to create and apply a Gaussian encoder configurable with gin.

  This is a separate function so that several different models (such as
  BetaVAE and FactorVAE) can call this function while the gin binding always
  stays 'encoder.(...)'. This makes it easier to configure models and parse
  the results files.

  Args:
    input_tensor: Tensor with image that should be encoded.
    is_training: Boolean that indicates whether we are training (usually
      required for batch normalization).
    num_latent: Integer with dimensionality of latent space.
    encoder_fn: Function that that takes the arguments (input_tensor,
      num_latent, is_training) and returns the tuple (means, log_vars) with the
      encoder means and log variances.

  Returns:
    Tuple (means, log_vars) with the encoder means and log variances.
  """
  with tf.variable_scope("encoder"):
    return encoder_fn(
        input_tensor=input_tensor,
        num_latent=num_latent,
        is_training=is_training)


@gin.configurable("decoder", whitelist=["decoder_fn"])
def make_decoder(latent_tensor,
                 output_shape,
                 is_training=True,
                 decoder_fn=gin.REQUIRED):
  """Gin wrapper to create and apply a decoder configurable with gin.

  This is a separate function so that several different models (such as
  BetaVAE and FactorVAE) can call this function while the gin binding always
  stays 'decoder.(...)'. This makes it easier to configure models and parse
  the results files.

  Args:
    latent_tensor: Tensor latent space embeddings to decode from.
    output_shape: Tuple with the output shape of the observations to be
      generated.
    is_training: Boolean that indicates whether we are training (usually
      required for batch normalization).
    decoder_fn: Function that that takes the arguments (input_tensor,
      output_shape, is_training) and returns the decoded observations.

  Returns:
    Tensor of decoded observations.
  """
  with tf.variable_scope("decoder"):
    return decoder_fn(
        latent_tensor=latent_tensor,
        output_shape=output_shape,
        is_training=is_training)


@gin.configurable("discriminator", whitelist=["discriminator_fn"])
def make_discriminator(input_tensor,
                       is_training=False,
                       discriminator_fn=gin.REQUIRED):
  """Gin wrapper to create and apply a discriminator configurable with gin.

  This is a separate function so that several different models (such as
  FactorVAE) can potentially call this function while the gin binding always
  stays 'discriminator.(...)'. This makes it easier to configure models and
  parse the results files.

  Args:
    input_tensor: Tensor on which the discriminator operates.
    is_training: Boolean that indicates whether we are training (usually
      required for batch normalization).
    discriminator_fn: Function that that takes the arguments
    (input_tensor, is_training) and returns tuple of (logits, clipped_probs).

  Returns:
    Tuple of (logits, clipped_probs) tensors.
  """
  with tf.variable_scope("discriminator"):
    logits, probs = discriminator_fn(input_tensor, is_training=is_training)
    clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
  return logits, clipped


@gin.configurable("fc_encoder", whitelist=[])
def fc_encoder(input_tensor, num_latent, is_training=True):
  """Fully connected encoder used in beta-VAE paper for the dSprites data.

  Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl).

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training

  flattened = tf.layers.flatten(input_tensor)
  e1 = tf.layers.dense(flattened, 1200, activation=tf.nn.relu, name="e1")
  e2 = tf.layers.dense(e1, 1200, activation=tf.nn.relu, name="e2")
  means = tf.layers.dense(e2, num_latent, activation=None)
  log_var = tf.layers.dense(e2, num_latent, activation=None)
  return means, log_var


@gin.configurable("conv_encoder", whitelist=[])
def conv_encoder(input_tensor, num_latent, is_training=True):
  """Convolutional encoder used in beta-VAE paper for the chairs data.

  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training

  e1 = tf.layers.conv2d(
      inputs=input_tensor,
      filters=32,
      kernel_size=4,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
      name="e1",
  )
  e2 = tf.layers.conv2d(
      inputs=e1,
      filters=32,
      kernel_size=4,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
      name="e2",
  )
  e3 = tf.layers.conv2d(
      inputs=e2,
      filters=64,
      kernel_size=2,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
      name="e3",
  )
  e4 = tf.layers.conv2d(
      inputs=e3,
      filters=64,
      kernel_size=2,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
      name="e4",
  )
  flat_e4 = tf.layers.flatten(e4)
  e5 = tf.layers.dense(flat_e4, 256, activation=tf.nn.relu, name="e5")
  means = tf.layers.dense(e5, num_latent, activation=None, name="means")
  log_var = tf.layers.dense(e5, num_latent, activation=None, name="log_var")
  return means, log_var

'''
begin caps
'''
@gin.configurable("caps_encoder", whitelist=[])
def caps_encoder(input_tensor, num_latent, is_training=True):
  """Capsul encoder used in capsule_VAE and Capsule_beta-VAE 

  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training

  input_image = Input(shape=(None, None, 1), name='capsule_input')
  Flat_input=Flatten(name='input_flattener')(input_image)

  # Layer 1: Just a conventional Conv2D layer
  conv1 = Conv2D(256, (9,9), activation='relu', name='conv1')(input_image)

  """now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
  then connect a Capsule layer.

  the output of final model is the lengths of 10 Capsule, whose dim=16.

  the length of Capsule is the proba,
  so the problem becomes a 10 two-classification problem.
  """

  conv_out = Reshape((-1, 256))(conv1)
  capsule = Capsule(10, 16, 3, True)(conv_out)


  # Here we mask the digit caps by the true label
  masked_by_y = Mask(name='mask_layer')([capsule, label_layer])
  #masked_by_y_flat=Flatten()(masked_by_y)


  outputCapsule = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
  rsh=Flatten()(masked_by_y)
  x = Dense(intermediate_dim, activation='relu')(rsh)
  z_mean = Dense(latent_dim, name='z_mean')(x)
  z_log_var = Dense(latent_dim, name='z_log_var')(x)

  # use reparameterization trick to push the sampling out as input
  # note that "output_shape" isn't necessary with the TensorFlow backend
  z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

  # instantiate encoder model
  encoder = Model([input_image, label_layer], [z_mean, z_log_var, z], name='encoder')
  encoder.summary()
'''
  flat_e4 = tf.layers.flatten(e4)
  e5 = tf.layers.dense(flat_e4, 256, activation=tf.nn.relu, name="e5")
  means = tf.layers.dense(e5, num_latent, activation=None, name="means")
  log_var = tf.layers.dense(e5, num_latent, activation=None, name="log_var")
  return means, log_var
'''
  '''
  end of caps
  '''

@gin.configurable("fc_decoder", whitelist=[])
def fc_decoder(latent_tensor, output_shape, is_training=True):
  """Fully connected encoder used in beta-VAE paper for the dSprites data.

  Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    latent_tensor: Input tensor to connect decoder to.
    output_shape: Shape of the data.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    Output tensor of shape (None, 64, 64, num_channels) with the [0,1] pixel
    intensities.
  """
  del is_training
  d1 = tf.layers.dense(latent_tensor, 1200, activation=tf.nn.tanh)
  d2 = tf.layers.dense(d1, 1200, activation=tf.nn.tanh)
  d3 = tf.layers.dense(d2, 1200, activation=tf.nn.tanh)
  d4 = tf.layers.dense(d3, np.prod(output_shape))
  return tf.reshape(d4, shape=[-1] + output_shape)


@gin.configurable("deconv_decoder", whitelist=[])
def deconv_decoder(latent_tensor, output_shape, is_training=True):
  """Convolutional decoder used in beta-VAE paper for the chairs data.

  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    latent_tensor: Input tensor of shape (batch_size,) to connect decoder to.
    output_shape: Shape of the data.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
      pixel intensities.
  """
  del is_training
  d1 = tf.layers.dense(latent_tensor, 256, activation=tf.nn.relu)
  d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
  d2_reshaped = tf.reshape(d2, shape=[-1, 4, 4, 64])
  d3 = tf.layers.conv2d_transpose(
      inputs=d2_reshaped,
      filters=64,
      kernel_size=4,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
  )

  d4 = tf.layers.conv2d_transpose(
      inputs=d3,
      filters=32,
      kernel_size=4,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
  )

  d5 = tf.layers.conv2d_transpose(
      inputs=d4,
      filters=32,
      kernel_size=4,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
  )
  d6 = tf.layers.conv2d_transpose(
      inputs=d5,
      filters=output_shape[2],
      kernel_size=4,
      strides=2,
      padding="same",
  )
  return tf.reshape(d6, [-1] + output_shape)


@gin.configurable("fc_discriminator", whitelist=[])
def fc_discriminator(input_tensor, is_training=True):
  """Fully connected discriminator used in FactorVAE paper for all datasets.

  Based on Appendix A page 11 "Disentangling by Factorizing"
  (https://arxiv.org/pdf/1802.05983.pdf)

  Args:
    input_tensor: Input tensor of shape (None, num_latents) to build
      discriminator on.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    logits: Output tensor of shape (batch_size, 2) with logits from
      discriminator.
    probs: Output tensor of shape (batch_size, 2) with probabilities from
      discriminator.
  """
  del is_training
  flattened = tf.layers.flatten(input_tensor)
  d1 = tf.layers.dense(flattened, 1000, activation=tf.nn.leaky_relu, name="d1")
  d2 = tf.layers.dense(d1, 1000, activation=tf.nn.leaky_relu, name="d2")
  d3 = tf.layers.dense(d2, 1000, activation=tf.nn.leaky_relu, name="d3")
  d4 = tf.layers.dense(d3, 1000, activation=tf.nn.leaky_relu, name="d4")
  d5 = tf.layers.dense(d4, 1000, activation=tf.nn.leaky_relu, name="d5")
  d6 = tf.layers.dense(d5, 1000, activation=tf.nn.leaky_relu, name="d6")
  logits = tf.layers.dense(d6, 2, activation=None, name="logits")
  probs = tf.nn.softmax(logits)
  return logits, probs


@gin.configurable("test_encoder", whitelist=["num_latent"])
def test_encoder(input_tensor, num_latent, is_training):
  """Simple encoder for testing.

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training
  flattened = tf.layers.flatten(input_tensor)
  means = tf.layers.dense(flattened, num_latent, activation=None, name="e1")
  log_var = tf.layers.dense(flattened, num_latent, activation=None, name="e2")
  return means, log_var


@gin.configurable("test_decoder", whitelist=[])
def test_decoder(latent_tensor, output_shape, is_training=False):
  """Simple decoder for testing.

  Args:
    latent_tensor: Input tensor to connect decoder to.
    output_shape: Output shape.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
      pixel intensities.
  """
  del is_training
  output = tf.layers.dense(latent_tensor, np.prod(output_shape), name="d1")
  return tf.reshape(output, shape=[-1] + output_shape)

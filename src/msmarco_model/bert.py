# This code is for tensorflow 2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
from msmarco_model.bert_common_v2 import *


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def get_config(self):
        return self.to_dict()


class BertModelInterface(object):
    def __init__(self):
        self.pooled_output = None
        self.sequence_output = None
        self.all_encoder_layers = None
        self.embedding_output = None
        self.embedding_table = None

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


class BertModel(BertModelInterface):
  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               scope=None):
      super(BertModel, self).__init__()
      config = copy.deepcopy(config)
      self.config = config
      if not is_training:
          config.hidden_dropout_prob = 0.0
          config.attention_probs_dropout_prob = 0.0

      input_shape = get_shape_list(input_ids, expected_rank=2)
      batch_size = input_shape[0]
      seq_length = input_shape[1]

      if input_mask is None:
          input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

      if token_type_ids is None:
          token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

      with tf.compat.v1.variable_scope(scope, default_name="bert"):
          with tf.compat.v1.variable_scope("embeddings"):
              # Perform embedding lookup on the word ids.
              (self.embedding_output, self.embedding_table) = embedding_lookup(
                  input_ids=input_ids,
                  vocab_size=config.vocab_size,
                  embedding_size=config.hidden_size,
                  initializer_range=config.initializer_range,
                  word_embedding_name="word_embeddings",
                  use_one_hot_embeddings=use_one_hot_embeddings)

              # Add positional embeddings and token type embeddings, then layer
              # normalize and perform dropout.
              self.embedding_output = embedding_postprocessor(
                  input_tensor=self.embedding_output,
                  use_token_type=True,
                  token_type_ids=token_type_ids,
                  token_type_vocab_size=config.type_vocab_size,
                  token_type_embedding_name="token_type_embeddings",
                  use_position_embeddings=True,
                  position_embedding_name="position_embeddings",
                  initializer_range=config.initializer_range,
                  max_position_embeddings=config.max_position_embeddings,
                  dropout_prob=config.hidden_dropout_prob)

          with tf.compat.v1.variable_scope("encoder"):
              attention_mask = create_attention_mask_from_input_mask(
                  input_ids, input_mask)

              self.all_encoder_layers = transformer_model(
                  input_tensor=self.embedding_output,
                  attention_mask=attention_mask,
                  input_mask=input_mask,
                  hidden_size=config.hidden_size,
                  num_hidden_layers=config.num_hidden_layers,
                  num_attention_heads=config.num_attention_heads,
                  is_training=is_training,
                  intermediate_size=config.intermediate_size,
                  intermediate_act_fn=get_activation(config.hidden_act),
                  hidden_dropout_prob=config.hidden_dropout_prob,
                  attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                  initializer_range=config.initializer_range,
                  do_return_all_layers=True)

          self.sequence_output = self.all_encoder_layers[-1]
          with tf.compat.v1.variable_scope("pooler"):
              first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
              self.pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                                         activation=tf.keras.activations.tanh,
                                                         kernel_initializer=create_initializer(config.initializer_range))(
                  first_token_tensor)


class BertModelMoreFF(BertModel):
  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               scope=None):
      super(BertModelMoreFF, self).__init__(
               config,
               is_training,
               input_ids,
               input_mask,
               token_type_ids,
               use_one_hot_embeddings,
               scope)

  def get_pooled_output(self):
      h1_dim = 2048
      h2_dim = 2048
      h1 = tf.keras.layers.Dense(h1_dim, activation=tf.keras.activations.tanh,
                                 kernel_initializer=create_initializer(self.config.initializer_range))(self.pooled_output)
      h2 = tf.keras.layers.Dense(h1_dim, activation=tf.keras.activations.tanh,
                                 kernel_initializer=create_initializer(self.config.initializer_range))(
          h1)
      h3 = tf.keras.layers.Dense(self.config.hidden_size, activation=tf.keras.activations.tanh,
                                 kernel_initializer=create_initializer(self.config.initializer_range))(
          h2)
      return h3



def transformer_model(input_tensor,
                    attention_mask=None,
                    input_mask=None,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    intermediate_act_fn=gelu,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    initializer_range=0.02,
                    is_training=True,
                    do_return_all_layers=False):
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    initializer = create_initializer(initializer_range)

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.compat.v1.variable_scope("attention"):
                attention_heads = []
                with tf.compat.v1.variable_scope("self"):
                    attention_head = attention_layer(
                            from_tensor=layer_input,
                            to_tensor=layer_input,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.compat.v1.variable_scope("output"):
                    attention_output = dense(hidden_size, initializer)(attention_output)
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.compat.v1.variable_scope("intermediate"):
                intermediate_output = dense(intermediate_size, initializer,
                                            activation=intermediate_act_fn)(attention_output)

            # Down-project back to `hidden_size` then add the residual.
            with tf.compat.v1.variable_scope("output"):
                layer_output = dense(hidden_size, initializer)(intermediate_output)
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def mimic_pooling(sequence_output, hidden_size, initializer_range):
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    pooled_output = tf.keras.layers.Dense(hidden_size,
                                          activation=tf.keras.activations.tanh,
                                          kernel_initializer=create_initializer(initializer_range))(
        first_token_tensor)
    return pooled_output
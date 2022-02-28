import tensorflow as tf
from typing import List, Union


def parse_args_initializer(initializer: Union[str, float, list]
                           ) -> tf.keras.initializers.Initializer:
    """Check process initializer input arguments"""
    if isinstance(initializer, str):
        return tf.keras.initializers.get(initializer)
    elif isinstance(initializer, (int, float, list)):
        return tf.keras.initializers.Constant(initializer)
    elif isinstance(initializer, tf.keras.initializers.Initializer):
        return initializer
    else:
        return None


class MultiLabelEmbedding(tf.keras.layers.Layer):
    """ Multi-label embedding for a fixed number of labels per data point

    Examples:
    ---------
    inputs = [[1, 2, 4], [0, 1, 2], [2, 1, 4], [3, 2, 1]]
    inputs = tf.constant(inputs)
    layer = MultiLabelEmbedding(
        vocab_size=500000, embed_size=300, random_state=42)
    y = layer(inputs)
    """
    def __init__(self,
                 vocab_size: int = None,
                 embed_size: int = None,
                 random_state: int = None,
                 initializer: Union[str, float, list] = 'ones',
                 regularizer: str = None,
                 constraint: str = None,
                 **kwargs):
        super(MultiLabelEmbedding, self).__init__(**kwargs)
        # store hyper params
        self.vocab_size = vocab_size   # v
        self.embed_size = embed_size   # e
        # other settings
        self.initializer = parse_args_initializer(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)
        # initialize layer weights
        if random_state:
            tf.random.set_seed(random_state)

    def build(self, input_shape=None):
        self.emb = self.add_weight(
            shape=(self.vocab_size, self.embed_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            trainable=self.trainable,
            name='multi_label_embedding')

    def call(self, inputs):
        h = tf.nn.embedding_lookup(params=self.emb, ids=inputs)
        h = tf.math.reduce_sum(h, axis=-2)
        return h

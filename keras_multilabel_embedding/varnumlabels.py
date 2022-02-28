import tensorflow as tf 
from .fixednumlabels import parse_args_initializer
from typing import Union


class MultiLabelEmbedding2(tf.keras.layers.Layer):
    """ Multi-label embedding for a variable number of labels per data point

    Example:
    --------
    inputs = [[1, 2, 4], [0, 1, 2], [2, 1], [3]] 
    layer = MultiLabelEmbedding2(
        vocab_size=500000, embed_size=300, random_state=42,
        initializer=tf.keras.initializers.VarianceScaling(seed=42))
    layer.build()
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
        super(MultiLabelEmbedding2, self).__init__(**kwargs)
        # store hyper params
        self.vocab_size = vocab_size   # v
        self.embed_size = embed_size   # e
        # other settings
        self.initializer = parse_args_initializer(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)
        # initialize layer weights
        if random_state:
            tf.keras.utils.set_random_seed(random_state)

    def build(self, input_shape=None):
        self.emb = self.add_weight(
            shape=(self.vocab_size, self.embed_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            trainable=self.trainable,
            name='multi_label_embedding')

    def call(self, inputs):
        hid = []
        for inp in inputs:
            h = tf.nn.embedding_lookup(params=self.emb, ids=inp)
            h = tf.math.reduce_sum(h, axis=-2)
            hid.append(h)
        return tf.stack(hid)

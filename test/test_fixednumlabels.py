from keras_multilabel_embedding import MultiLabelEmbedding
import tensorflow as tf


def test1():
    # the lookup based Embedding
    x_ids = [[1, 2, 4], [0, 1, 2], [2, 1, 4], [3, 2, 1]]
    x_ids = tf.constant(x_ids)
    layer1 = MultiLabelEmbedding(
        vocab_size=5, embed_size=300, random_state=42,
        initializer=tf.keras.initializers.VarianceScaling(seed=42))
    layer1.build()
    y1 = layer1(x_ids)
    # the linear layer
    x_ids = [[0., 1, 1, 0, 1], [1, 1, 1, 0, 0],
             [0, 1, 1, 0, 1], [0, 1, 1, 1, 0]]
    x_ids = tf.constant(x_ids)
    layer2 = tf.keras.layers.Dense(300, use_bias=False)
    layer2.build(input_shape=(None, 5))
    layer2.set_weights(layer1.get_weights())
    y2 = layer2(x_ids)
    # compare
    tf.debugging.assert_near(y1, y2)

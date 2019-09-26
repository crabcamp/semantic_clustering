import os

import tensorflow as tf
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SentenceEncoder:
    def __init__(self, model_path):
        self.model_path = model_path

    def _load(self):
        if hasattr(self, '_encode'):
            return

        with tf.compat.v1.Graph().as_default():
            encoder = hub.Module(self.model_path)
            encoder_input = tf.compat.v1.placeholder(tf.string)
            embeddings = encoder(encoder_input)
            session = tf.compat.v1.train.MonitoredSession()

            def encode(sentences):
                return session.run(embeddings, {encoder_input: sentences})

            self._encode = encode

    def encode(self, sentences):
        self._load()
        return self._encode(sentences)

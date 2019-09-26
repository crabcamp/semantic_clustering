import os

import tensorflow as tf
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class KeytermsVectorizer:
    def __init__(self, model_path):
        self.model_path = model_path

    def _load(self):
        if hasattr(self, '_vectorizer'):
            return

        with tf.Graph().as_default():
            sentences = tf.compat.v1.placeholder(tf.string)
            embed = hub.Module(self.model_path)
            embeddings = embed(sentences)
            session = tf.compat.v1.train.MonitoredSession()

            def vectorize(phrases):
                return session.run(embeddings, {sentences: phrases})

            self._vectorizer = vectorize

    def vectorize(self, keyterms):
        self._load()

        return self._vectorizer(keyterms)

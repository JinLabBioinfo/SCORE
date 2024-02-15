import sys
import traceback
import numpy as np
from score.experiments.experiment import Experiment


def reparameterize(mean, logvar):
    import tensorflow as tf
    eps = tf.random.normal(shape=tf.shape(mean))
    return eps * tf.exp(logvar * .5) + mean


def MultivariateNormalTriL_loader(latent_dim):
    import tensorflow_probability as tfp
    def load_MultivariateNormalTriL(name, trainable, dtype,
                                    function, function_type, module,
                                    output_shape, output_shape_type,
                                    output_shape_module, arguments,
                                    make_distribution_fn, convert_to_tensor_fn):
        return tfp.layers.MultivariateNormalTriL(latent_dim, name=name,
                                                 trainable=trainable, dtype=dtype,
                                                 convert_to_tensor_fn=convert_to_tensor_fn)

    return load_MultivariateNormalTriL


class VaDEExperiment(Experiment):
    def __init__(self, name, x, y, depths, data_generator, encoder_file=None, encoder=None, **kwargs):
        from tensorflow import keras
        super().__init__(name, x, y, depths, data_generator, **kwargs)
        #assert encoder_file is not None or encoder is not None, "Must supply either encoder file or encoder"
        if encoder_file:
            try:
                with open(encoder_file.replace('.h5', '.json'), 'r') as f:
                    self.encoder = keras.models.model_from_json(f.read())
                self.encoder.load_weights(encoder_file)
            except FileNotFoundError:
                custom_objects = {"MultivariateNormalTriL": MultivariateNormalTriL_loader(16)}
                self.encoder = keras.models.load_model(encoder_file, custom_objects=custom_objects)
        else:
            self.encoder = encoder

    def get_embedding(self, iter_n=0, batch_size=32):
        if 'load_vade_from' in self.other_args.keys() and self.other_args['load_vade_from'] is not None:
                embedding = np.load(f"vade/{self.dataset_name}/0/embedding_{self.other_args['load_vade_from']}.npy")
                print(embedding.shape)
        else:
            import tensorflow as tf
            embedding = None
            for batch_i in range(0, self.x.shape[0], batch_size):
                tmp = self.encoder(self.x[batch_i: batch_i + batch_size]).mean()
                if embedding is None:
                    embedding = tmp
                else:
                    embedding = tf.concat([embedding, tmp], axis=0)
        return embedding

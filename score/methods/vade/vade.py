import os
import time
import joblib
import wandb
import random
import argparse
import matplotlib
matplotlib.use('Agg')  # necessary when plotting without $DISPLAY
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import tqdm
from tensorflow_probability import distributions as tfd
from wandb.integration.keras import WandbCallback




def make_mixture_prior(latent_size, mixture_components, distribution='normal'):
    """Creates the mixture of Gaussians prior distribution.
  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.
  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
    if distribution == 'normal':
        loc = tf.compat.v1.get_variable(
            name="loc", shape=[mixture_components, latent_size])
        raw_scale_diag = tf.compat.v1.get_variable(
            name="raw_scale_diag", shape=[mixture_components, latent_size])
        mixture_logits = tf.compat.v1.get_variable(
            name="mixture_logits", shape=[mixture_components])

        return tfd.MixtureSameFamily(
            components_distribution=tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=tf.nn.softplus(raw_scale_diag), name='prior'),
            mixture_distribution=tfd.Categorical(logits=mixture_logits, name='prior_categorical'),
            name="prior_mixture")

    elif distribution == 'laplace':
        loc = tf.compat.v1.get_variable(
            name="loc", shape=[mixture_components, latent_size])
        raw_scale_diag = tf.compat.v1.get_variable(
            name="raw_scale_diag", shape=[mixture_components, latent_size])
        mixture_logits = tf.compat.v1.get_variable(
            name="mixture_logits", shape=[mixture_components])

        return tfd.MixtureSameFamily(
            components_distribution=tfd.Independent(tfd.Laplace(
                loc=loc,
                scale=tf.nn.softplus(raw_scale_diag)), name='prior'),
            mixture_distribution=tfd.Categorical(logits=mixture_logits, name='prior_categorical'),
            name="prior_mixture")

    elif distribution == 't':
        df = tf.compat.v1.get_variable(
            name="df", shape=[mixture_components])
        loc = tf.compat.v1.get_variable(
            name="loc", shape=[mixture_components, latent_size])
        raw_scale = tf.Variable(initial_value=tf.ones((mixture_components, latent_size, latent_size)),
            name="raw_scale")
        mixture_logits = tf.compat.v1.get_variable(
            name="mixture_logits", shape=[mixture_components])

        return tfd.MixtureSameFamily(
            components_distribution=tfd.MultivariateStudentTLinearOperator(
                df=df,
                loc=loc,
                scale=tf.linalg.LinearOperatorFullMatrix(tf.nn.softplus(raw_scale))),
            mixture_distribution=tfd.Categorical(logits=mixture_logits, name='prior_categorical'),
            name="prior")


def train_vade(features, dataset, experiment, run_i, args, preprocessing=None, load_results=False, wandb_config=None):
    from score.methods.vade.sc_vade_callback import VisualizeCallback
    start_time = time.time()
    if args.wandb:
        wandb.login(key='')
    plt.style.use('default')

    distribution = 'normal'
    beta = args.beta
    config = wandb.config

    n_clusters = args.n_clusters
    if n_clusters is None:
        n_clusters = len(np.unique(dataset.reference['cluster']))
        if n_clusters == 1:
            n_clusters = 10  # reasonable default
    n_cell_types = len(np.unique(dataset.reference['cluster']))
    start_filters = args.start_filters
    start_filter_size = args.start_filter_size
    stride = args.stride
    stride_y = args.stride_y
    latent_dim = args.latent_dim
    n_epochs = args.n_epochs
    batch_size = dataset.n_cells if args.batch_size == -1 else args.batch_size
    learning_rate = args.lr

    vade_config={'n_clusters': n_clusters,
                       'beta': beta,
                       'learning_rate': learning_rate,
                       'batch_size': batch_size,
                       'stride': stride,
                       'start_filters': start_filters,
                       'start_filter_size': start_filter_size,
                       'latent_dim': latent_dim,
                       'model_depth': dataset.depth,
                       'resolution': dataset.resolution,
                       'n_strata': dataset.limit2Mb,
                       'strata_offset': dataset.rotated_offset,
                       'binary': dataset.binarize,
                       'min_depth': args.min_depth,
                       'downsample': args.downsample,
                       'stride_y': stride_y}
    if wandb_config is None:
        wandb_config = {}
    wandb_config = {**wandb_config, **vade_config}
    if not load_results:
        sparse_matrices = {}
        if preprocessing is None:
            sparse_matrices_filename = f'sparse_matrices_{dataset.dataset_name}_{dataset.res_name}.sav'
        else:
            sparse_matrices_filename = f'sparse_matrices_{dataset.dataset_name}_{dataset.res_name}_{"_".join(preprocessing)}.sav'
        os.makedirs('data/sparse_matrices', exist_ok=True)
        if sparse_matrices_filename in os.listdir('data/sparse_matrices'):
            print('Loading sparse matrix data...')
            with open(os.path.join('data/sparse_matrices', sparse_matrices_filename), 'rb') as f:
                sparse_matrices = joblib.load(f)

        model_depth = dataset.depth
        if start_filter_size != 3:
            model_depth_effective = model_depth + int(((int((start_filter_size - 1) / 2)) / 2))  # first layer can reduce dims by more than double
            print(model_depth_effective)
            if dataset.active_idxs is None:
                matrix_len = int(len(dataset.anchor_list))
            else:
                matrix_len = len(dataset.active_idxs)
            next = matrix_len + (2 ** model_depth_effective  - matrix_len % (2 ** model_depth_effective))
            dataset.matrix_pad = int(next - matrix_len)

        x_train = []
        y = []
        depths = []
        batches = []
        cellnames = []
        #print(matrix_len, dataset.matrix_len, dataset.matrix_pad)
        for cell_i in tqdm(range(dataset.n_cells)):
            cell, label, depth, batch, cellname = dataset.get_cell_by_index(cell_i, downsample=False, preprocessing=preprocessing)
            if cell is None:  # cell had no reads
                continue
            x_train.append(cell)
            y.append(label)
            depths.append(depth)
            batches.append(batch)
            cellnames.append(cellname)
        if args.binarize:
            x_train = np.array(x_train, dtype=bool)
        else:
            x_train = np.array(x_train, dtype='float32')
        try:
            with open(os.path.join('data/sparse_matrices', sparse_matrices_filename), 'wb') as f:
                joblib.dump(dataset.sparse_matrices, f)  # and save the sparse matrix dict for use later
        except MemoryError:
            print('Not enough memory to save')

        print(x_train.shape)
        if preprocessing is not None:
            if 'idf' in preprocessing:
                from sklearn.preprocessing import normalize
                x_flat = x_train.reshape(x_train.shape[0], -1)
                idf = np.log(x_flat.shape[0] / (np.sum(x_flat > 0, axis=0) + 1))
                x_flat = x_flat * idf
                x_flat = np.nan_to_num(x_flat)
                x_flat = normalize(x_flat, norm="l2")
                x_train = x_flat.reshape(x_train.shape)
                args.gaussian_output = True  # model normalized idf weights using gaussian

        y = np.array(y)
        depths = np.array(depths)
        batches = np.array(batches)
        batches = batches - batches.min()  # zero-indexed batches are required for batch removal

        if dataset.active_idxs is None:
            input_shape = (dataset.limit2Mb, dataset.matrix_len + dataset.matrix_pad, 1)
        else:
            input_shape = (dataset.limit2Mb, len(dataset.active_idxs) + dataset.matrix_pad, 1)

        print(input_shape)

        input_layer = tf.keras.Input(input_shape)
        h = input_layer
        for i in range(model_depth):
            still_2d = (input_shape[0] / (2 ** i)) > 1
            h = tf.keras.layers.Conv2D(filters=start_filters * (i + 1),
                                    kernel_size=start_filter_size if i == 0 else 3,
                                    strides=(int((start_filter_size - 1) / 2) if stride_y and still_2d else 1, int((start_filter_size - 1) / 2)) if i == 0 else (stride if stride_y and still_2d else 1, stride),
                                    # kernel_regularizer=keras.regularizers.l1(regulizer_alpha),
                                    # strides=(1 if i > 2 else 2, 2),
                                    padding='same')(h)
            h = tf.keras.layers.LeakyReLU(0.2)(h)
            h = tf.keras.layers.Conv2D(filters=start_filters * (i + 1),
                                    kernel_size=3,
                                    strides=1,
                                    # kernel_regularizer=keras.regularizers.l1(regulizer_alpha),
                                    padding='same')(h)
            h = tf.keras.layers.LeakyReLU(0.2)(h)
            # h = keras.layers.MaxPooling2D((2, 2))(h)

        h_shape = list(tf.keras.backend.int_shape(h)[1:])
        print(h_shape)
        if stride_y:
            h_shape[0] = int(h_shape[0] / (stride / (int((start_filter_size - 1) / 2))))
        h_shape[1] = int(h_shape[1] / (stride / (int((start_filter_size - 1) / 2))))
        h_shape = tuple(h_shape)
        print(h_shape)
        h = tf.keras.layers.Flatten()(h)
        params_size = tfp.layers.MultivariateNormalTriL.params_size(latent_dim)
        x = tf.keras.layers.Dense(params_size)(h)
        gm_layer = tfp.layers.MultivariateNormalTriL(latent_dim,
                                                    activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                                                        make_mixture_prior(latent_dim, n_clusters,
                                                                            distribution=distribution), weight=beta))

        output = gm_layer(x)

        encoder = tf.keras.Model(input_layer, output, name='encoder')
        experiment.encoder = encoder
        experiment.x = x_train
        experiment.y = np.squeeze(np.asarray(y).ravel())

        latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
        h = tf.keras.layers.Dense(np.prod(h_shape))(latent_inputs)
        h = tf.keras.layers.Reshape(h_shape)(h)

        for i in range(model_depth):
            still_2d = (input_shape[0] / (2 ** (model_depth - i - 1))) > 1
            n_filters = start_filters * (model_depth - i)
            h = tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                            kernel_size=3,
                                            strides=1,
                                            padding='same')(h)
            h = tf.keras.layers.LeakyReLU(0.2)(h)
            h = tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                            kernel_size=3,
                                            strides=(stride if stride_y and still_2d else 1, stride),
                                            # strides=(2 if i > 2 else 1, 2),
                                            padding='same')(h)
            h = tf.keras.layers.LeakyReLU(0.2)(h)

        
        if args.gaussian_output:
            h = tf.keras.layers.Conv2D(2, 1)(h)
            h = tf.keras.layers.Flatten()(h)
            x_recon = tfp.layers.IndependentNormal(input_shape)(h)
        else:
            h = tf.keras.layers.Conv2D(1, 1)(h)
            h = tf.keras.layers.Flatten()(h)
            if dataset.binarize:
                x_recon = tfp.layers.IndependentBernoulli(input_shape, tfd.Bernoulli.logits)(h)
            else:
                x_recon = tfp.layers.IndependentPoisson(input_shape)(h)

        decoder = tf.keras.models.Model(latent_inputs, x_recon, name='decoder')

        vae = tf.keras.Model(inputs=encoder.inputs,
                        outputs=decoder(encoder.outputs))

        print(vae.summary())

        loss = lambda x, rv_x: -rv_x.log_prob(x)
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=args.weight_decay
            )
        except Exception as e:
            print(e)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            )
        vae.compile(optimizer=optimizer,
                    loss=loss)


        viz_callback = VisualizeCallback(x_train, y, encoder, decoder, vae, gm_layer, dataset, n_clusters,
                                        n_cell_types, features, n_epochs, batch_size,
                                        ll_norm = np.prod(input_shape), log_wandb=args.wandb,
                                        save_interval=100, update_interval=100,
                                        save_dir='vade/%s/%d/' % (dataset.dataset_name, run_i),
                                        model_dir='vade_models/%s/' % dataset.dataset_name)

        logdir = 'logs/%s' % dataset.dataset_name
        os.makedirs(logdir, exist_ok=True)

        if args.wandb:
            wandb_callback = WandbCallback(log_gradients=False)
        else:
            wandb_callback = None
        
        if dataset.downsample:
            _ = vae.fit(dataset,
                        epochs=n_epochs,
                        callbacks=[viz_callback],
                        batch_size=batch_size,
                        verbose=2)
        else:
            _ = vae.fit(x_train, x_train,
                        epochs=n_epochs,
                        callbacks=[viz_callback],
                        batch_size=batch_size,
                        verbose=2)

    
    experiment.run(load=load_results, outer_iter=run_i, start_time=start_time, log_wandb=args.wandb, wandb_config=wandb_config)

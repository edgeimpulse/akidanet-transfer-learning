import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import shutil, datetime, json, time, threading, sys, os
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, BatchNormalization,
    Conv2D, GlobalMaxPooling2D, Lambda, GlobalAveragePooling2D)
from tensorflow.keras.models import Sequential, Model
import ei_tensorflow.training

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# This creates NN embeddings, will use tSNE if <=5000 samples, or PCA if larger than 5000
def create_embeddings(base_model, dir_path, out_file_x):
    start_time = time.time()

    print('Creating embeddings...')

    dr_complete = False

    def still_saving_main():
        time.sleep(2)
        while not dr_complete:
            print('Still creating embeddings...', flush=True)
            time.sleep(2)

    progress_thread = threading.Thread(target=still_saving_main)
    progress_thread.start()

    try:
        SHAPE = tuple(base_model.layers[0].get_input_at(0).get_shape().as_list()[1:])

        x_file = os.path.join(dir_path, 'X_train_features.npy')

        # large file? then mmap
        if (os.path.getsize(x_file) > 256 * 1024 * 1024):
            rows = None
            with open(x_file, 'rb') as npy:
                version = np.lib.format.read_magic(npy)
                shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                rows = shape[0]

            X = np.memmap(x_file, shape=tuple([ rows ]) + SHAPE)
        # otherwise just load into memory
        else:
            X = np.load(x_file)
            X = X.reshape(tuple([ X.shape[0] ]) + SHAPE)

        model = Sequential()
        model.add(InputLayer(input_shape=SHAPE, name='x_input'))
        model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output))
        model.add(Flatten())

        X_pred = model.predict(X)

        if (X.shape[0] > 5000):
            print('WARN: More than 5000 samples, using PCA to create embeddings')
            scaler = StandardScaler()
            X_pred = scaler.fit_transform(X_pred)

            pca = PCA(n_components=2, random_state=3)
            dr_res = pca.fit_transform(X_pred)
        else:
            tsne = TSNE(2, learning_rate='auto', init='pca')
            dr_res = tsne.fit_transform(X_pred)

        np.save(out_file_x, np.ascontiguousarray(dr_res))
        dr_complete = True
        print('Creating embeddings OK (took ' + str(round(time.time() - start_time)) + ' seconds)')
        print('')
    except Exception as e:
        dr_complete = True
        print('WARN: Creating embeddings failed:', e)
        print('')

import sys, os, shutil, signal, random, operator, functools, time, subprocess, math, contextlib, io, skimage, argparse
import logging, threading

dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Edge Impulse training scripts')
parser.add_argument('--info-file', type=str, required=False,
                    help='train_input.json file with info about classes and input shape',
                    default=os.path.join(dir_path, 'train_input.json'))
parser.add_argument('--data-directory', type=str, required=True,
                    help='Where to read the data from')
parser.add_argument('--out-directory', type=str, required=True,
                    help='Where to write the data')

parser.add_argument('--epochs', type=int, required=False,
                    help='Number of training cycles')
parser.add_argument('--learning-rate', type=float, required=False,
                    help='Learning rate')

args, unknown = parser.parse_known_args()

# Info about the training pipeline (inputs / shapes / modes etc.)
if not os.path.exists(args.info_file):
    print('Info file', args.info_file, 'does not exist')
    exit(1)

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import numpy as np

# Suppress Numpy deprecation warnings
# TODO: Only suppress warnings in production, not during development
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# Filter out this erroneous warning (https://stackoverflow.com/a/70268806 for context)
warnings.filterwarnings('ignore', 'Custom mask layers require a config and must override get_config')

RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

# Since it also includes TensorFlow and numpy, this library should be imported after TensorFlow has been configured
sys.path.append('./resources/libraries')
import ei_tensorflow.training
import ei_tensorflow.conversion
import ei_tensorflow.profiling
import ei_tensorflow.inference
import ei_tensorflow.embeddings
import ei_tensorflow.lr_finder
import ei_tensorflow.brainchip.model
from ei_shared.parse_train_input import parse_train_input, parse_input_shape


import json, datetime, time, traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error

BEST_MODEL_PATH = os.path.join(os.sep, 'tmp', 'best_model.hdf5')

input = parse_train_input(args.info_file)

# Information about the data and input:
# The shape of the model's input (which may be different from the shape of the data)
MODEL_INPUT_SHAPE = parse_input_shape(input.inputShapeString)
# The length of the model's input, used to determine the reshape inside the model
MODEL_INPUT_LENGTH = MODEL_INPUT_SHAPE[0]
MAX_TRAINING_TIME_S = input.maxTrainingTimeSeconds

online_dsp_config = None

if (online_dsp_config != None):
    print('The online DSP experiment is enabled; training will be slower than normal.')

# load imports dependening on import
if (input.mode == 'object-detection' and input.objectDetectionLastLayer == 'mobilenet-ssd'):
    import ei_tensorflow.object_detection

def exit_gracefully(signum, frame):
    print("")
    print("Terminated by user", flush=True)
    time.sleep(0.2)
    sys.exit(1)


def train_model(train_dataset, validation_dataset, input_length, callbacks,
                X_train, X_test, Y_train, Y_test, train_sample_count, classes, classes_values):
    global ei_tensorflow

    override_mode = None
    disable_per_channel_quantization = False
    # We can optionally output a Brainchip Akida pre-trained model
    akida_model = None

    if (input.mode == 'object-detection' and input.objectDetectionLastLayer == 'mobilenet-ssd'):
        ei_tensorflow.object_detection.set_limits(max_training_time_s=MAX_TRAINING_TIME_S,
            is_enterprise_project=input.isEnterpriseProject)

    
    
    MODEL_INPUT_SHAPE = (160, 160, 3,)
    learning_rate = 0.0005
    epochs = 20
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, InputLayer, Activation, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed
    from tensorflow.keras.optimizers import Adam
    
    from keras import Model
    from akida_models.layer_blocks import dense_block
    
    from akida_models import akidanet_imagenet
    from keras.utils.data_utils import get_file
    
    base_model = akidanet_imagenet(input_shape=MODEL_INPUT_SHAPE,
                                   classes=classes,
                                   alpha=0.5,
                                   include_top=False,
                                   pooling='avg',
                                   weight_quantization=4,
                                   activ_quantization=4,
                                   input_weight_quantization=8)
    
    pretrained_weights = get_file(
        "akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5",
        by_name=True)
    
    base_model.load_weights(pretrained_weights, by_name=True)
    base_model.trainable = False
    
    from keras import Model
    from keras.layers import Activation, Dropout, Reshape, Flatten
    from akida_models.layer_blocks import dense_block
    
    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = dense_block(x,
                    units=16,
                    name='fc1',
                    add_batchnorm=False,
                    add_activation=True)
    x = Dropout(0.5, name='dropout_1')(x)
    x = dense_block(x,
                    units=classes,
                    name='predictions',
                    add_batchnorm=False,
                    add_activation=False)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape')(x)
    
    model = Model(base_model.input, x, name='')
    
    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        learning_rate,
        epochs,
        end_learning_rate=learning_rate / 10.0,
        power=1.0,
        cycle=False,
        name=None,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    BATCH_SIZE = 32
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
    validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)
    
    model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=2, callbacks=callbacks)
    
    print('')
    print('Initial training done.', flush=True)
    print('')
    
    fine_tune_loss = 'categorical_crossentropy'
    fine_tune_metrics = ['accuracy']
    from ei_tensorflow import training
    import cnn2snn
    
    model = training.load_best_model(BEST_MODEL_PATH, akida_model=True)
    
    #! Remove any dropout from the model, since it makes quantization-aware training hard
    model = training.remove_dropout(model)
    
    print('Performing post-training quantization...')
    
    akida_model = cnn2snn.quantize(model,
                               weight_quantization=4,
                               activ_quantization=4,
                               input_weight_quantization=8)
    print('Performing post-training quantization OK')
    print('')
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      mode='max',
                                                      verbose=1,
                                                      min_delta=0,
                                                      patience=10,
                                                      restore_best_weights=True)
    
    print('Fine-tuning to recover accuracy...')
    akida_model.compile(optimizer=opt,
                    loss=fine_tune_loss,
                    metrics=fine_tune_metrics)
    
    akida_model.fit(train_dataset,
                    epochs=30,
                    verbose=2,
                    validation_data=validation_dataset,
                    callbacks=[early_stopping]
                )
    
    print('Fine-tuning to recover accuracy OK')
    print('')
    
    return model, override_mode, disable_per_channel_quantization, akida_model

# This callback ensures the frontend doesn't time out by sending a progress update every interval_s seconds.
# This is necessary for long running epochs (in big datasets/complex models)
class BatchLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, train_sample_count, epochs, interval_s = 10):
        # train_sample_count could be smaller than the batch size, so make sure total_batches is atleast
        # 1 to avoid a 'divide by zero' exception in the 'on_train_batch_end' callback.
        self.total_batches = max(1, int(train_sample_count / batch_size))
        self.last_log_time = time.time()
        self.epochs = epochs
        self.interval_s = interval_s

    # Within each epoch, print the time every 10 seconds
    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        if self.last_log_time + self.interval_s < current_time:
            print('Epoch {0}% done'.format(int(100 / self.total_batches * batch)), flush=True)
            self.last_log_time = current_time

    # Reset the time the start of every epoch
    def on_epoch_end(self, epoch, logs=None):
        self.last_log_time = time.time()

def main_function():
    """This function is used to avoid contaminating the global scope"""
    classes_values = input.classes
    classes = 1 if input.mode == 'regression' else len(classes_values)

    mode = input.mode
    object_detection_last_layer = input.objectDetectionLastLayer if input.mode == 'object-detection' else None

    train_dataset, validation_dataset, samples_dataset, X_train, X_test, Y_train, Y_test, has_samples, X_samples, Y_samples = ei_tensorflow.training.get_dataset_from_folder(
        input, args.data_directory, RANDOM_SEED, online_dsp_config, MODEL_INPUT_SHAPE
    )

    callbacks = ei_tensorflow.training.get_callbacks(mode, BEST_MODEL_PATH,
        object_detection_last_layer=object_detection_last_layer,
        is_enterprise_project=input.isEnterpriseProject,
        max_training_time_s=MAX_TRAINING_TIME_S)

    model = None

    print('')
    print('Training model...')
    print('Training on {0} inputs, validating on {1} inputs'.format(len(X_train), len(X_test)))
    # USER SPECIFIC STUFF
    model, override_mode, disable_per_channel_quantization, akida_model = train_model(train_dataset, validation_dataset,
        MODEL_INPUT_LENGTH, callbacks, X_train, X_test, Y_train, Y_test, len(X_train), classes, classes_values)
    if override_mode is not None:
        mode = override_mode
    # END OF USER SPECIFIC STUFF

    # REST OF THE APP
    print('Finished training', flush=True)
    print('', flush=True)

    # Make sure these variables are here, even when quantization fails
    tflite_quant_model = None

    if mode == 'object-detection':
        tflite_model, tflite_quant_model = ei_tensorflow.object_detection.convert_to_tf_lite(
            args.out_directory,
            saved_model_dir='saved_model',
            validation_dataset=validation_dataset,
            model_filenames_float='model.tflite',
            model_filenames_quantised_int8='model_quantized_int8_io.tflite')
    elif mode == 'segmentation':
        from ei_tensorflow.constrained_object_detection.conversion import convert_to_tf_lite
        tflite_model, tflite_quant_model = convert_to_tf_lite(
            args.out_directory, model,
            saved_model_dir='saved_model',
            h5_model_path='model.h5',
            validation_dataset=validation_dataset,
            model_filenames_float='model.tflite',
            model_filenames_quantised_int8='model_quantized_int8_io.tflite',
            disable_per_channel=disable_per_channel_quantization)
    else:
        model, tflite_model, tflite_quant_model = ei_tensorflow.conversion.convert_to_tf_lite(
            model, BEST_MODEL_PATH, args.out_directory,
            saved_model_dir='saved_model',
            h5_model_path='model.h5',
            validation_dataset=validation_dataset,
            model_input_shape=MODEL_INPUT_SHAPE,
            model_filenames_float='model.tflite',
            model_filenames_quantised_int8='model_quantized_int8_io.tflite',
            disable_per_channel=disable_per_channel_quantization,
            syntiant_target=input.syntiantTarget,
            akida_model=input.akidaModel)

        if input.akidaModel:
            if not akida_model:
                print('Akida training code must assign a quantized model to a variable named "akida_model"', flush=True)
                exit(1)
            ei_tensorflow.brainchip.model.convert_akida_model(args.out_directory, akida_model,
                                                              'akida_model.fbz',
                                                              MODEL_INPUT_SHAPE)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    main_function()
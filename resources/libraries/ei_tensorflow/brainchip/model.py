import os, shutil, time
import tensorflow as tf
import numpy as np

def convert_akida_model(dir_path, model, model_path, input_shape):
    from cnn2snn import convert
    import akida

    print('Converting to Akida model...')
    print('')
    # https://doc.brainchipinc.com/api_reference/cnn2snn_apis.html#convert
    # The input_scaling param works like this:
    # input_akida = input_scaling[0] * input_keras + input_scaling[1]
    # It needs to be matched by a similar conversion when we perform inference.
    input_is_image = False
    input_scaling = None
    if len(input_shape) == 3:
        input_is_image = True
        # Don't set input scaling explicitly if there is a rescaling layer present
        # TODO: Make an explicit function for doing these layer checks
        rescaling = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling)]
        if len(rescaling) == 0:
            input_scaling = (255, 0)

    model_akida = convert(model, input_is_image=input_is_image, input_scaling=input_scaling)

    model_akida.map(akida.AKD1000())

    model_akida.summary()
    print('Converting to Akida model OK')
    print('')

    print('Saving Akida model...')
    model_akida.save(os.path.join(dir_path, model_path))
    print('Saving Akida model OK...')

def save_akida_model(akida_model, path):
    print('Saving Akida model...', flush=True)
    to_save = tf.keras.models.clone_model(akida_model)
    to_save.save(path)
    print('Saving Akida model OK', flush=True)

def load_akida_model(path):
    import akida
    return akida.Model(path)

def make_predictions(mode, model_path, validation_dataset,
                    Y_test, train_dataset, Y_train, test_dataset, Y_real_test):

    if mode != 'classification':
        raise Exception('Only classification is supported')
    prediction_train = None
    prediction_test = None

    prediction = predict(model_path, validation_dataset, len(Y_test))
    if (train_dataset is not None) and (Y_train is not None):
        prediction_train = predict(model_path, train_dataset, len(Y_train))
    if (test_dataset is not None) and (Y_real_test is not None):
        prediction_test = predict(model_path, test_dataset, len(Y_real_test))
    return prediction, prediction_train, prediction_test

def predict(model_path, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    model = load_akida_model(model_path)

    last_log = time.time()

    pred_y = []
    for item, label in validation_dataset.take(-1).as_numpy_iterator():
        item = (item * 255).astype('uint8')
        item = np.expand_dims(item, axis=0)
        output = model.predict(item)
        output = np.squeeze(output)
        pred_y.append(output)
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)



from __future__ import print_function
import json
import time
import traceback
import os
import numpy as np
import tensorflow as tf
import json, datetime, time, traceback
import os, shutil, operator, functools, time, subprocess, math
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import log_loss, mean_squared_error
import ei_tensorflow.inference
import ei_tensorflow.brainchip.model

from ei_tensorflow.constrained_object_detection.util import batch_convert_segmentation_map_to_object_detection_prediction
from ei_tensorflow.constrained_object_detection.metrics import non_background_metrics
from ei_tensorflow.constrained_object_detection.metrics import dataset_match_by_near_centroids

def tflite_predict(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for item, label in validation_dataset.take(-1).as_numpy_iterator():
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        scores = ei_tensorflow.inference.process_output(output_details, output)
        pred_y.append(scores)
        # Print an update at least every 10 seconds
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)

def tflite_predict_object_detection(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
          item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
          item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
          interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
          interpreter.invoke()
          rect_label_scores = ei_tensorflow.inference.process_output_object_detection(output_details, interpreter)
          pred_y.append(rect_label_scores)
          # Print an update at least every 10 seconds
          current_time = time.time()
          if last_log + 10 < current_time:
              print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
              last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_yolov5(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
          item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
          item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
          interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
          interpreter.invoke()
          rect_label_scores = ei_tensorflow.inference.process_output_yolov5(output_details, interpreter)
          pred_y.append(rect_label_scores)
          # Print an update at least every 10 seconds
          current_time = time.time()
          if last_log + 10 < current_time:
              print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
              last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_segmentation(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    last_log = time.time()

    y_pred = []
    for item, _ in validation_dataset.take(-1):
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        output = ei_tensorflow.inference.process_output(output_details, output)
        y_pred.append(output)
        # Print an update at least every 10 seconds
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(y_pred) - 1))), flush=True)
            last_log = current_time

    y_pred = np.stack(y_pred)

    return y_pred

def get_tensor_details(tensor):
    """Obtains the quantization parameters for a given tensor"""
    details = {
        'dataType': None,
        'name': tensor['name'],
        'shape': tensor['shape'].tolist(),
        'quantizationScale': None,
        'quantizationZeroPoint': None
    }
    if tensor['dtype'] is np.int8:
        details['dataType'] = 'int8'
        details['quantizationScale'] = tensor['quantization'][0]
        details['quantizationZeroPoint'] = tensor['quantization'][1]
    elif tensor['dtype'] is np.float32:
        details['dataType'] = 'float32'
    else:
        raise Exception('Model tensor has an unknown datatype, ', tensor['dtype'])

    return details


def get_io_details(model, model_type):
    """Gets the input and output datatype and quantization details for a model"""
    interpreter = tf.lite.Interpreter(model_content=model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = list(map(get_tensor_details, input_details))
    outputs = list(map(get_tensor_details, output_details))

    return {
        'modelType': model_type,
        'inputs': inputs,
        'outputs': outputs
    }

def make_predictions(mode, model, validation_dataset, Y_test, train_dataset,
                            Y_train, test_dataset, Y_real_test, akida_model_path):
    if akida_model_path:
        # TODO: We should avoid vendor-specific naming at this level, for maintainability
        return ei_tensorflow.brainchip.model.make_predictions(mode, akida_model_path, validation_dataset,
                                                              Y_test, train_dataset, Y_train, test_dataset, Y_real_test)

    return make_predictions_tflite(mode, model, validation_dataset, Y_test,
                                   train_dataset, Y_train, test_dataset, Y_real_test)

def make_predictions_tflite(mode, model, validation_dataset, Y_test, train_dataset, Y_train, test_dataset, Y_real_test):
    prediction_train = None
    prediction_test = None

    if mode == 'object-detection':
        prediction = tflite_predict_object_detection(model, validation_dataset, len(Y_test))
    elif mode == 'yolov5':
        prediction = tflite_predict_yolov5(model, validation_dataset, len(Y_test))
    elif mode == 'segmentation':
        prediction = tflite_predict_segmentation(model, validation_dataset, len(Y_test))
    else:
        prediction = tflite_predict(model, validation_dataset, len(Y_test))
        if (not train_dataset is None) and (not Y_train is None):
            prediction_train = tflite_predict(model, train_dataset, len(Y_train))
        if (not test_dataset is None) and (not Y_real_test is None):
            prediction_test = tflite_predict(model, test_dataset, len(Y_real_test))

    return prediction, prediction_train, prediction_test

def profile_model(model_type, model, model_file, validation_dataset, Y_test, X_samples, Y_samples,
                         has_samples, memory, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                         num_classes, train_dataset=None, Y_train=None, test_dataset=None, Y_real_test=None,
                         akida_model_path=None):
    """Calculates performance statistics for a TensorFlow Lite model"""
    matrix_train=None
    matrix_test=None
    report_train=None
    report_test=None

    prediction, prediction_train, prediction_test = make_predictions(mode, model, validation_dataset, Y_test,
                                                                     train_dataset, Y_train, test_dataset,
                                                                     Y_real_test, akida_model_path)

    if mode == 'classification':
        Y_labels = []
        for ix in range(num_classes):
            Y_labels.append(ix)
        matrix = confusion_matrix(Y_test.argmax(axis=1), prediction.argmax(axis=1), labels=Y_labels)
        report = classification_report(Y_test.argmax(axis=1), prediction.argmax(axis=1), output_dict=True, zero_division=0)
        if not prediction_train is None:
            matrix_train = confusion_matrix(Y_train.argmax(axis=1), prediction_train.argmax(axis=1))
            report_train = classification_report(Y_train.argmax(axis=1), prediction_train.argmax(axis=1), output_dict=True, zero_division=0)
        if not prediction_test is None:
            matrix_test = confusion_matrix(Y_real_test.argmax(axis=1), prediction_test.argmax(axis=1))
            report_test = classification_report(Y_real_test.argmax(axis=1), prediction_test.argmax(axis=1), output_dict=True, zero_division=0)

        accuracy = report['accuracy']
        loss = log_loss(Y_test, prediction)
        try:
            # Make predictions for feature explorer
            if has_samples:
                if model:
                    feature_explorer_predictions = tflite_predict(model, X_samples, len(Y_samples))
                elif akida_model_path:
                    feature_explorer_predictions = ei_tensorflow.brainchip.model.predict(akida_model_path, X_samples, len(Y_samples))
                else:
                    raise Exception('Expecting either a Keras model or an Akida model')

                # Store each prediction with the original sample for the feature explorer
                prediction_samples = np.concatenate((Y_samples, np.array([feature_explorer_predictions.argmax(axis=1) + 1]).T), axis=1).tolist()
            else:
                prediction_samples = []
        except Exception as e:
            print('Failed to generate feature explorer', e, flush=True)
            prediction_samples = []
    elif mode == 'regression':
        matrix = np.array([])
        report = {}
        accuracy = 0
        loss = mean_squared_error(Y_test, prediction[:,0])
        try:
            # Make predictions for feature explorer
            if has_samples:
                feature_explorer_predictions = tflite_predict(model, X_samples, len(Y_samples))
                # Store each prediction with the original sample for the feature explorer
                prediction_samples = np.concatenate((Y_samples, feature_explorer_predictions), axis=1).tolist()
            else:
                prediction_samples = []
        except Exception as e:
            print('Failed to generate feature explorer', e, flush=True)
            prediction_samples = []
    elif mode == 'object-detection' or mode == 'yolov5':
        # This is only installed on object detection containers so import it only when used
        from mean_average_precision import MetricBuilder
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=num_classes)
        # Calculate mean average precision
        def un_onehot(onehot_array):
            """Go from our one-hot encoding to an index"""
            val = np.argmax(onehot_array, axis=0)
            return val
        for index, sample in enumerate(validation_dataset.take(-1).unbatch()):
            data = sample[0]
            labels = sample[1]
            p = prediction[index]
            gt = []
            curr_ps = []

            boxes = labels[0]
            labels = labels[1]
            for box_index, box in enumerate(boxes):
                label = labels[box_index]
                label = un_onehot(label)
                gt.append([box[0], box[1], box[2], box[3], label, 0, 0])

            for p2 in p:
                curr_ps.append([p2[0][0], p2[0][1], p2[0][2], p2[0][3], p2[1], p2[2]])

            gt = np.array(gt)
            curr_ps = np.array(curr_ps)

            metric_fn.add(curr_ps, gt)

        coco_map = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05),
                                   recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

        matrix = np.array([])
        report = {}
        accuracy = float(coco_map)
        loss = 0
        prediction_samples = []
    elif mode == 'segmentation':

        _batch, width, height, num_classes = prediction.shape
        if width != height:
            raise Exception("Expected segmentation output to be square, not",
                            prediction.shape)
        output_width_height = width

        # y_true has already been extracted during tflite_predict_segmentation
        # and has labels including implicit background class = 0

        # TODO(mat): what should minimum_confidence_rating be here?
        y_pred = batch_convert_segmentation_map_to_object_detection_prediction(
            prediction, minimum_confidence_rating=0.5, fuse=True)

        # do alignment by centroids. this results in a flatten list of int
        # labels that is suitable for confusion matrix calculations.
        y_true_labels, y_pred_labels = dataset_match_by_near_centroids(
            # batch the data since the function expects it
            validation_dataset.batch(32, drop_remainder=False), y_pred, output_width_height)

        # TODO(mat): we need to pass out recall too for FOMO
        matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=range(num_classes))
        _precision, _recall, f1 = non_background_metrics(y_true_labels, y_pred_labels, num_classes)
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
        accuracy = f1
        loss = 0
        prediction_samples = []

    if not memory:
        memory = calculate_memory(model_file, model_type, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script)

    model_size = 0
    if model:
        model_size = len(model)

    return {
        'type': model_type,
        'loss': loss,
        'accuracy': accuracy,
        'accuracyTrain': report_train['accuracy'] if not report_train is None else None,
        'accuracyTest': report_test['accuracy'] if not report_test is None else None,
        'confusionMatrix': matrix.tolist(),
        'confusionMatrixTrain': matrix_train.tolist() if not matrix_train is None else None,
        'confusionMatrixTest': matrix_test.tolist() if not matrix_test is None else None,
        'report': report,
        'reportTrain': report_train,
        'reportTest': report_test,
        'size': model_size,
        'estimatedMACCs': None,
        'memory': memory,
        'predictions': prediction_samples
    }

def calculate_memory(model_file, model_type, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script):
    if (mode == 'object-detection' or mode == 'yolov5'):
        memory = {}
        memory['tflite'] = {
            'ram': 0,
            'rom': os.stat(model_file).st_size,
            'arenaSize': 0,
            'modelSize': os.stat(model_file).st_size
        }
        memory['eon'] = {
            'ram': 0,
            'rom': os.stat(model_file).st_size,
            'arenaSize': 0,
            'modelSize': os.stat(model_file).st_size
        }
    # Some models don't have the scripts (e.g. akida) so skip this step
    elif prepare_model_tflite_script and prepare_model_tflite_eon_script:
        memory = {}

        try:
            with open('prepare_tflite.sh', 'w') as f:
                f.write(prepare_model_tflite_script(model_file))
            with open('prepare_eon.sh', 'w') as f:
                f.write(prepare_model_tflite_eon_script(model_file))

            print('Profiling ' + model_type + ' model (tflite)...', flush=True)

            if os.path.exists('/app/benchmark/tflite-model'):
                shutil.rmtree('/app/benchmark/tflite-model')
            subprocess.check_output(['sh', 'prepare_tflite.sh']).decode("utf-8")
            tflite_output = json.loads(subprocess.check_output(['/app/benchmark/benchmark.sh', '--tflite-type',
                'float32',
                '--tflite-file', model_file
                ]).decode("utf-8"))
            if os.getenv('K8S_ENVIRONMENT') == 'staging' or os.getenv('K8S_ENVIRONMENT') == 'test':
                print(tflite_output['logLines'])

            print('Profiling ' + model_type + ' model (EON)...', flush=True)

            if os.path.exists('/app/benchmark/tflite-model'):
                shutil.rmtree('/app/benchmark/tflite-model')
            subprocess.check_output(['sh', 'prepare_eon.sh']).decode("utf-8")
            eon_output = json.loads(subprocess.check_output(['/app/benchmark/benchmark.sh', '--tflite-type',
                'float32',
                '--tflite-file', model_file,
                '--eon'
                ]).decode("utf-8"))
            if os.getenv('K8S_ENVIRONMENT') == 'staging' or os.getenv('K8S_ENVIRONMENT') == 'test':
                print(eon_output['logLines'], flush=True)

            # Add fudge factor since the target architecture is different (only on TFLite)
            old_arena_size = tflite_output['arenaSize']
            extra_arena_size = int(math.floor((math.ceil(old_arena_size) * 0.2) + 1024))

            tflite_output['ram'] = tflite_output['ram'] + extra_arena_size
            tflite_output['arenaSize'] = tflite_output['arenaSize'] + extra_arena_size

            memory['tflite'] = tflite_output
            memory['eon'] = eon_output
        except Exception as err:
            print('Error while finding memory:', flush=True)
            print(err, flush=True)
            traceback.print_exc()
            memory = None
    else:
        memory = None

    return memory

# Useful reference: https://machinethink.net/blog/how-fast-is-my-model/
def estimate_maccs_for_layer(layer):
    """Estimate the number of multiply-accumulates in a given Keras layer."""
    """Better than flops because there's hardware support for maccs."""
    if isinstance(layer, tf.keras.layers.Dense):
        # Ignore the batch dimension
        input_count = functools.reduce(operator.mul, layer.input.shape[1:], 1)
        return input_count * layer.units

    if (isinstance(layer, tf.keras.layers.Conv1D)
        or isinstance(layer, tf.keras.layers.Conv2D)
        or isinstance(layer, tf.keras.layers.Conv3D)):
        kernel_size = functools.reduce(operator.mul, layer.kernel_size)
        # The channel is either at the start or the end of the shape (ignoring)
        # the batch dimension
        if layer.data_format == 'channels_first':
            input_channels = layer.input.shape[1]
        else:
            input_channels = layer.input.shape[-1]
        # Ignore the batch dimension but include the channels
        output_size = functools.reduce(operator.mul, layer.output.shape[1:])
        return kernel_size * input_channels * output_size

    if (isinstance(layer, tf.keras.layers.SeparableConv1D)
        or isinstance(layer, tf.keras.layers.SeparableConv1D)
        or isinstance(layer, tf.keras.layers.DepthwiseConv2D)):
        kernel_size = functools.reduce(operator.mul, layer.kernel_size)
        if layer.data_format == 'channels_first':
            input_channels = layer.input.shape[1]
            output_channels = layer.output.shape[1]
            # Unlike regular conv, don't include the channels
            output_size = functools.reduce(operator.mul, layer.output.shape[2:])
        else:
            input_channels = layer.input.shape[-1]
            output_channels = layer.output.shape[-1]
            # Unlike regular conv, don't include the channels
            output_size = functools.reduce(operator.mul, layer.output.shape[1:-1])
        # Calculate the MACCs for depthwise and pointwise steps
        depthwise_count = kernel_size * input_channels * output_size
        # If this is just a depthwise conv, we can return early
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return depthwise_count
        # Otherwise, calculate MACCs for the pointwise step and add them
        pointwise_count = input_channels * output_size * output_channels
        return depthwise_count + pointwise_count

    if isinstance(layer, tf.keras.Model):
        return estimate_maccs_for_model(layer)

    # For other layers just return 0. These are mostly stuff that doesn't involve MACCs
    # or stuff that isn't supported by TF Lite for Microcontrollers yet.
    return 0

def estimate_maccs_for_model(keras_model):
    maccs = 0
    for layer in keras_model.layers:
        try:
            layer_maccs = estimate_maccs_for_layer(layer)
            maccs += layer_maccs
        except Exception as err:
            print('Error while estimating maccs for layer', flush=True)
            print(err, flush=True)
    return maccs

def describe_layers(keras_model):
    layers = []
    for l in range(len(keras_model.layers)):
        layer = keras_model.layers[l]
        input = layer.input
        if isinstance(input, list):
            input = input[0]
        layers.append({
            'input': {
                'shape': input.shape[1],
                'name': input.name,
                'type': str(input.dtype)
            },
            'output': {
                'shape': layer.output.shape[1],
                'name': layer.output.name,
                'type': str(layer.output.dtype)
            }
        })

    return layers


def get_recommended_model_type(float32_perf, int8_perf):
    # For now, always recommend int8 if available
    if int8_perf:
        return 'int8'
    else:
        return 'float32'

def get_model_metadata(keras_model, validation_dataset, Y_test, X_samples, Y_samples, has_samples,
                       class_names, curr_metadata, mode, prepare_model_tflite_script,
                       prepare_model_tflite_eon_script, model_float32=None, model_int8=None,
                       file_float32=None, file_int8=None, file_akida=None,
                       train_dataset=None, Y_train=None, test_dataset=None, Y_real_test=None):

    metadata = {
        'metadataVersion': 5,
        'created': datetime.datetime.now().isoformat(),
        'classNames': class_names,
        'availableModelTypes': [],
        'recommendedModelType': '',
        'modelValidationMetrics': [],
        'modelIODetails': [],
        'mode': mode,
        'kerasJSON': None,
        'performance': None
    }

    recalculate_memory = True
    recalculate_performance = True

    # For some model types (e.g. object detection) there is no keras model, so
    # we are unable to compute some of our stats with these methods
    if keras_model:
        # This describes the basic inputs and outputs, but skips over complex parts
        # such as transfer learning base models
        metadata['layers'] = describe_layers(keras_model)
        estimated_maccs = estimate_maccs_for_model(keras_model)
        # This describes the full model, so use it to determine if the architecture
        # has changed between runs
        metadata['kerasJSON'] = keras_model.to_json()
        # Only recalculate memory when model architecture has changed
        if (
            curr_metadata and 'kerasJSON' in curr_metadata and 'metadataVersion' in curr_metadata
            and curr_metadata['metadataVersion'] == metadata['metadataVersion']
            and metadata['kerasJSON'] == curr_metadata['kerasJSON']
        ):
            recalculate_memory = False
        else:
            recalculate_memory = True

        if (
            curr_metadata and 'kerasJSON' in curr_metadata and 'metadataVersion' in curr_metadata
            and curr_metadata['metadataVersion'] == metadata['metadataVersion']
            and metadata['kerasJSON'] == curr_metadata['kerasJSON']
            and 'performance' in curr_metadata
            and curr_metadata['performance']
        ):
            metadata['performance'] = curr_metadata['performance']
            recalculate_performance = False
        else:
            recalculate_memory = True
            recalculate_performance = True
    else:
        metadata['layers'] = []
        estimated_maccs = -1
        # If there's no Keras model we can't tell if the architecture has changed, so recalculate memory every time
        recalculate_memory = True
        recalculate_performance = True

    if recalculate_performance:
        try:
            args = '/app/profiler/build/profiling '
            if file_float32:
                args = args + file_float32 + ' '
            if file_int8:
                args = args + file_int8 + ' '

            print('Calculating inferencing time...', flush=True)
            a = os.popen(args).read()
            metadata['performance'] = json.loads(a[a.index('{'):a.index('}')+1])
            print('Calculating inferencing time OK', flush=True)
        except Exception as err:
            print('Error while calculating inferencing time:', flush=True)
            print(err, flush=True)
            traceback.print_exc()
            metadata['performance'] = None

    float32_perf = None
    int8_perf = None

    if model_float32:
        try:
            print('Profiling float32 model...', flush=True)
            model_type = 'float32'

            memory = None
            if not recalculate_memory:
                curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_metadata['modelValidationMetrics']))
                if (len(curr_metrics) > 0):
                    memory = curr_metrics[0]['memory']

            float32_perf = profile_model(model_type, model_float32, file_float32, validation_dataset, Y_test, X_samples, Y_samples, has_samples, memory, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script, len(class_names), train_dataset, Y_train, test_dataset, Y_real_test)
            float32_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(float32_perf)
            metadata['modelIODetails'].append(get_io_details(model_float32, model_type))
        except Exception as err:
            print('Unable to execute TensorFlow Lite float32 model:', flush=True)
            print(err, flush=True)
            traceback.print_exc()

    if model_int8:
        try:
            print('Profiling int8 model...', flush=True)
            model_type = 'int8'

            memory = None
            if not recalculate_memory:
                curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_metadata['modelValidationMetrics']))
                if (len(curr_metrics) > 0):
                    memory = curr_metrics[0]['memory']

            int8_perf = profile_model(model_type, model_int8, file_int8, validation_dataset, Y_test, X_samples, Y_samples, has_samples, memory, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script, len(class_names), train_dataset, Y_train, test_dataset, Y_real_test)
            int8_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(int8_perf)
            metadata['modelIODetails'].append(get_io_details(model_int8, model_type))
        except Exception as err:
            print('Unable to execute TensorFlow Lite int8 model:', flush=True)
            print(err, flush=True)
            traceback.print_exc()

    if file_akida:
        print('Profiling akida model...', flush=True)
        model_type = 'akida'
        memory = None
        akida_perf = profile_model(model_type, None, None, validation_dataset, Y_test, X_samples,
                                   Y_samples, has_samples, memory, mode, None,
                                   None, len(class_names), train_dataset,
                                   Y_train, test_dataset, Y_real_test, file_akida)
        metadata['availableModelTypes'].append(model_type)
        metadata['modelValidationMetrics'].append(akida_perf)

    # Decide which model to recommend
    if file_akida:
        metadata['recommendedModelType'] = 'akida'
    else:
        recommended_model_type = get_recommended_model_type(float32_perf, int8_perf)
        metadata['recommendedModelType'] = recommended_model_type

    return metadata

def profile_tflite_file(file, mode, prepare_model_tflite_script, prepare_model_tflite_eon_script):
    metadata = {
        'metadataVersion': 1
    }
    try:
        args = '/app/profiler/build/profiling ' + file

        print('Calculating inferencing time...', flush=True)
        a = os.popen(args).read()
        metadata['performance'] = json.loads(a[a.index('{'):a.index('}')+1])
        print('Calculating inferencing time OK', flush=True)
    except Exception as err:
        print('Error while calculating inferencing time:', flush=True)
        print(err, flush=True)
        traceback.print_exc()
        metadata['performance'] = None

    metadata['memory'] = calculate_memory(file, 'model', mode, prepare_model_tflite_script, prepare_model_tflite_eon_script)
    return metadata

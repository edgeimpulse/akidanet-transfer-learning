import numpy as np
import tensorflow as tf
import os, json, time
from typing_extensions import Literal

from tensorflow.lite.python.interpreter import Interpreter

import ei_tensorflow.utils
from ei_tensorflow.constrained_object_detection.util import convert_segmentation_map_to_object_detection_prediction
from ei_tensorflow.constrained_object_detection.util import convert_sample_bbox_and_labels_to_boundingboxlabelscores
from ei_tensorflow.constrained_object_detection.metrics import non_background_metrics
from ei_tensorflow.constrained_object_detection.metrics import match_by_near_centroids

def process_input(input_details, data):
    """Prepares an input for inference, quantizing if necessary.

    Args:
        input_details: The result of calling interpreter.get_input_details()
        data (numpy array): The raw input data

    Returns:
        A tensor object representing the input, quantized if necessary
    """
    if input_details[0]['dtype'] is np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        data = (data / scale) + zero_point
        data = np.around(data)
        data = data.astype(np.int8)
    return tf.convert_to_tensor(data)

def process_output(output_details, output) -> 'list[float]':
    """Transforms an output tensor into a Python list, dequantizing if necessary.

    Args:
        output_details: The result of calling interpreter.get_output_details()
        data (tensor): The raw output tensor

    Returns:
        A Python list representing the output, dequantized if necessary
    """
    # If the output tensor is int8, dequantize the output data
    if output_details[0]['dtype'] is np.int8:
        scale = output_details[0]['quantization'][0]
        zero_point = output_details[0]['quantization'][1]
        output = output.astype(np.float32)
        output = (output - zero_point) * scale
    return output[0].tolist()

def process_output_yolov5(output_details, interpreter, minimum_confidence_rating=None):
    """Transforms an output tensor into a Python list for object detection
    models.
    Args:
        output_details: The result of calling interpreter.get_output_details()
        interpreter: The interpreter

    Returns:
        A Python list representing the output
    """

    output_data = interpreter.get_tensor(output_details[0]['index'])

    xyxy, classes, scores = yolov5_detect(output_data) #boxes(x,y,x,y), classes(int), scores(float) [25200]

    rects = []
    labels = []
    score_res = []

    if (minimum_confidence_rating == None):
        minimum_confidence_rating = 0.01

    for i in range(len(scores)):
        if ((scores[i] >= minimum_confidence_rating) and (scores[i] <= 1.0)):
            xmin = float(xyxy[0][i])
            ymin = float(xyxy[1][i])
            xmax = float(xyxy[2][i])
            ymax = float(xyxy[3][i])

            # Who in their right min has decided to do ymin,xmin,ymax,xmax ?
            bbox = [ymin, xmin, ymax, xmax]

            rects.append(bbox)
            labels.append(int(classes[i]))
            score_res.append(float(scores[i]))

    raw_scores = list(zip(rects, labels, score_res))
    return raw_scores

def process_output_object_detection(output_details, interpreter, minimum_confidence_rating=None):
    """Transforms an output tensor into a Python list for object detection
    models.
    Args:
        output_details: The result of calling interpreter.get_output_details()
        interpreter: The interpreter

    Returns:
        A Python list representing the output
    """
    # Models trained before and after our TF2.7 upgrade have different output tensor orders;
    # use names instead of indices to ensure correct functioning.
    # Create a map of name to output_details index:
    name_map = {o['name']: o['index'] for o in output_details}
    # StatefulPartitionedCall:0 is number of detections, which we can ignore
    try:
        scores = interpreter.get_tensor(name_map['StatefulPartitionedCall:1'])[0].tolist()
        labels = interpreter.get_tensor(name_map['StatefulPartitionedCall:2'])[0].tolist()
        rects = interpreter.get_tensor(name_map['StatefulPartitionedCall:3'])[0].tolist()
    except KeyError:
        # If the expected names are missing, default to legacy order
        scores = interpreter.get_tensor(output_details[2])[0].tolist()
        labels = interpreter.get_tensor(output_details[1])[0].tolist()
        rects = interpreter.get_tensor(output_details[0])[0].tolist()

    combined = list(zip(rects, labels, scores))

    # Filter out any scores that don't meet the minimum
    if minimum_confidence_rating is not None:
        return list(filter(lambda x: x[2] >= minimum_confidence_rating, combined))
    else:
        return combined

def compute_performance_object_detection(raw_scores: list, width_height: int,
                                         y_data: dict, num_classes: int):

    info = {
        'sampleId': y_data['sampleId'],
    }
    if len(raw_scores) > 0:
        info['boxes'], info['labels'], info['scores'] = list(zip(*raw_scores))
    else:
        info['boxes'], info['labels'], info['scores'] = [], [], []

    # If there are no ground truth bounding boxes, emit either a perfect or a zero score
    if len(y_data['boundingBoxes']) == 0:
        if len(raw_scores) == 0:
            info['mAP'] = 1
        else:
            info['mAP'] = 0
        return info

    def convert_y_box_format(box):
        coords = ei_tensorflow.utils.convert_box_coords(box, width_height)
        return [coords[0], coords[1], coords[2], coords[3], int(box['label'] - 1), 0, 0]

    def convert_preds_format(p):
        return [p[0][0], p[0][1], p[0][2], p[0][3], int(p[1]), p[2]]

    gt = np.array(list(map(convert_y_box_format, y_data['boundingBoxes'])))
    preds = np.array(list(map(convert_preds_format, raw_scores)))

    # This is only installed on object detection containers so import it only when used
    from mean_average_precision import MetricBuilder
    metric_fn_pred = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)
    metric_fn_pred.add(preds, gt)
    metric_pred = metric_fn_pred.value(iou_thresholds=np.arange(0.5, 1.0, 0.05),
                                         recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')
    coco_map_pred = metric_pred['mAP']

    # The mAP calculation is designed to run over an entire dataset, so it expects all classes
    # to be present. However, for a given image it is common that only some classes are present.
    # For a classifier trained on 2 classes and an image with only 1, the maximum mAP is 1 / 2.
    # For a classifier trained on 3 classes and an image with only 1, the maximum mAP is 1 / 3.
    # To make up for this, we should divide the actual mAP by the maximum mAP for that image.
    classes_in_gt = len(set([box['label'] for box in y_data['boundingBoxes']]))
    maximum_mAP = classes_in_gt / num_classes
    scaled_mAP = coco_map_pred / maximum_mAP
    info['mAP'] = float(scaled_mAP)

    return info

def run_model(mode: Literal['classification', 'regression', 'object-detection', 'segmentation', 'yolov5'], interpreter: Interpreter,
              item: np.ndarray, specific_input_shape: 'list[int]', minimum_confidence_rating: float=None,
              y_data=None, num_classes: int=None):
    """Runs inference with a given model and mode
    """
    if mode == 'object-detection':
        return run_object_detection_inference(interpreter, item, specific_input_shape,
                                  minimum_confidence_rating, y_data, num_classes)
    elif mode == 'segmentation':
        return run_segmentation_inference(interpreter, item, specific_input_shape,
                                  minimum_confidence_rating, y_data)
    elif mode == 'yolov5':
        return run_yolov5_inference(interpreter, item, specific_input_shape,
                                    minimum_confidence_rating, y_data, num_classes)
    elif mode == 'classification' or mode == 'regression':
        return run_vector_inference(interpreter, item, specific_input_shape)
    else:
        raise ValueError('Invalid mode "' + mode + '"')

def invoke(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]'):
    """Invokes the Python TF Lite interpreter with a given input
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    item_as_tensor = process_input(input_details, item)
    if specific_input_shape:
        item_as_tensor = tf.reshape(item_as_tensor, specific_input_shape)
    # Add batch dimension
    item_as_tensor = tf.expand_dims(item_as_tensor, 0)
    interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output, output_details

def run_vector_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]'):
    """Runs inference that produces a vector output (classification or regression)
    """
    output, output_details = invoke(interpreter, item, specific_input_shape)
    scores = process_output(output_details, output)
    return scores

def run_object_detection_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                                   minimum_confidence_rating: float, y_data, num_classes):
    """Runs inference that produces an object detection output
    """
    width, height, _channels = specific_input_shape
    if width != height:
        raise Exception(f"Only square inputs are supported; not {specific_input_shape}")
    width_height = width

    output, output_details = invoke(interpreter, item, specific_input_shape)
    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not num_classes:
        raise ValueError('num_classes must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')
    raw_scores = process_output_object_detection(output_details, interpreter, minimum_confidence_rating)
    scores = compute_performance_object_detection(raw_scores, width_height, y_data, num_classes)
    return scores

def run_segmentation_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                               minimum_confidence_rating: float, y_data: list):
    """Runs inference that produces an object detection output
    """

    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')

    width, height, _channels = specific_input_shape
    if width != height:
        raise Exception(f"Only square input is supported; not {specific_input_shape}")
    input_width_height = width

    output, output_details = invoke(interpreter, item, specific_input_shape)
    output = ei_tensorflow.inference.process_output(output_details, output)
    output = np.array(output)

    _batch, width, height, num_classes_including_background = output_details[0]['shape']
    if width != height:
        raise Exception(f"Only square output is supported; not {output_details[0]['shape']}")
    output_width_height = width

    # convert y_true to list of BoundingBoxLabelScores. note: this data is
    # 1 indexed already so covers the class=0 for implicit background
    y_true_boxes_labels_scores = convert_sample_bbox_and_labels_to_boundingboxlabelscores(
        y_data['boundingBoxes'], input_width_height)

    # convert model output to list of BoundingBoxLabelScores including fusing
    # of adjacent boxes. retains class=0 from segmentation output.
    y_pred_boxes_labels_scores = convert_segmentation_map_to_object_detection_prediction(
        output, minimum_confidence_rating, fuse=True)

    # do alignment by centroids
    y_true_labels, y_pred_labels = match_by_near_centroids(
        y_true_boxes_labels_scores, y_pred_boxes_labels_scores,
        output_width_height=output_width_height,
        min_normalised_distance=0.2,
        return_debug_info=False)

    _precision, _recall, f1 = non_background_metrics(
        y_true_labels, y_pred_labels,
        num_classes_including_background)

    # package up into info dict
    # as final step to return to studio map labels by -1 to remove class=0
    # background class.
    # TODO(mat) this is not actually a mAP score :/
    # TODO(mat) also pass back recall
    boxes = [list(bls.bbox) for bls in y_pred_boxes_labels_scores]
    labels = [bls.label-1 for bls in y_pred_boxes_labels_scores]
    scores = [bls.score for bls in y_pred_boxes_labels_scores]
    return {
        'sampleId': y_data['sampleId'],
        'boxes': boxes, 'labels': labels, 'scores': scores,
        'mAP': f1
    }

def yolov5_class_filter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def yolov5_detect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = yolov5_class_filter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

def run_yolov5_inference(interpreter: Interpreter, item: np.ndarray, specific_input_shape: 'list[int]',
                           minimum_confidence_rating: float, y_data: list, num_classes):
    """Runs inference that produces an object detection output
    """
    if not y_data:
        raise ValueError('y_data must be provided for object detection')
    if not minimum_confidence_rating:
        raise ValueError('minimum_confidence_rating must be provided for object detection')

    width, height, _channels = specific_input_shape
    if width != height:
        raise Exception(f"Only square input is supported; not {specific_input_shape}")
    input_width_height = width

    output, output_details = invoke(interpreter, item, specific_input_shape)

    raw_scores = process_output_yolov5(output_details, interpreter, minimum_confidence_rating)
    return compute_performance_object_detection(raw_scores, input_width_height, y_data, num_classes)

def prepare_interpreter(dir_path, model_path, multithreading=False):
    """Instantiates an interpreter, allocates its tensors, and returns it."""
    lite_file_path = os.path.join(dir_path, os.path.basename(model_path))
    num_threads = None
    if multithreading:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
        print(f'Using {num_threads} threads for inference.', flush=True)
    interpreter = tf.lite.Interpreter(model_path=lite_file_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter

def map_test_label_to_train(test_ix, train_labels, test_labels, zero_index=True):
    """Converts a test label index to an index relevant to the original set of training labels"""

    # For FOMO we work with 1-indexed labels, but the original label map is always 0-indexed
    adjust_index = 0 if zero_index else 1

    actual_label = test_labels[test_ix - adjust_index]

    # Test label not in training labels? Use an out-of-range index.
    # These label indices are only used for profiling results in Python.
    # The studio only sees the original set of labels, so any index not present in the training set
    # is fine here.
    if actual_label not in train_labels:
        return -1
    return train_labels.index(actual_label) + adjust_index

def classify_keras(input_x_file, input_y_file, mode, output_file, dir_path,
                   model_path, specific_input_shape, use_tflite, layer_input_name,
                   layer_output_name, class_names_training, class_names_testing,
                   minimum_confidence_rating):

    gt_y = None

    input = np.load(input_x_file, mmap_mode='r')
    if (not isinstance(input[0], (np.ndarray))):
        input = np.array([ input ])

    if ei_tensorflow.utils.is_y_structured(os.path.join(dir_path, 'y_classify.npy')):
        gt_y = ei_tensorflow.utils.load_y_structured(dir_path, 'y_classify.npy', len(input))
        for row in gt_y:
            for box in row['boundingBoxes']:
                # Studio is passing label indices using the testing dataset
                # This is ok for other models as we don't profile in Python; we just feed-back
                # the raw results and map to the correct labels later.
                # For structured data we profile in Python so we need the correct labels.
                box['label'] = map_test_label_to_train(box['label'],
                    class_names_training, class_names_testing, False)

    # Predictions array
    pred_y = []

    # Make sure we log every 10 seconds
    LOG_MIN_INTERVAL_S = 10
    last_log_time = time.time()
    showed_slow_warning = False

    # In this code path, we use a TensorFlow Lite model
    if use_tflite:
        interpreter = prepare_interpreter(dir_path, model_path)
        for i, item in enumerate(input):
            scores = run_model(mode, interpreter, item,
                               specific_input_shape, minimum_confidence_rating,
                               gt_y[i] if gt_y != None else None, len(class_names_training))
            pred_y.append(scores)
            # Log a message if enough time has elapsed
            current_time = time.time()
            if last_log_time + LOG_MIN_INTERVAL_S < current_time:
                message = '{0}% done'.format(int(100 / len(input) * i))
                if not showed_slow_warning:
                    message += ' (this can take a while for large datasets)'
                    showed_slow_warning = True
                print(message, flush=True)
                last_log_time = current_time

    # Otherwise, we can expect a 1.13 .pb file, and we need to do more complex things
    # to use it.
    else:
        # See https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
        def wrap_frozen_graph(graph_def, inputs, outputs):
            def _imports_graph_def():
                tf.compat.v1.import_graph_def(graph_def, name="")
            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        graph_def = tf.compat.v1.GraphDef()
        pb_file_path = os.path.join(dir_path, 'trained.pb')
        graph_def.ParseFromString(open(pb_file_path, 'rb').read())
        model_func = wrap_frozen_graph(
            graph_def, inputs=layer_input_name,
            outputs=layer_output_name)

        for item in input:
            item_as_tensor = tf.convert_to_tensor(item)
            item_as_tensor = tf.expand_dims(item_as_tensor, 0)
            output = model_func(item_as_tensor)
            scores = output[0].numpy().tolist()
            pred_y.append(scores)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(json.dumps(pred_y, separators=(',', ':')))
    else:
        print('Begin output')
        print(json.dumps(pred_y, separators=(',', ':')))
        print('End output')
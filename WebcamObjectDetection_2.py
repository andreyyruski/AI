import cv2
import numpy as np
import os
import time
import tensorflow as tf
from utils import visualization_utils as vis_util
from utils import label_map_util


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

## Loading label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# What model to download.
#MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"
#MODEL_NAME = "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03"
#MODEL_NAME = "ssdlite_mobilenet_v2_coco_2018_05_09"
#MODEL_NAME = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
#MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"

# TRT Models
#MODEL_NAME = "ssd_inception_v2_coco_trt"
MODEL_NAME = "faster_rcnn_resnet50_coco_trt"
#MODEL_NAME = "ssd_resnet_50_fpn_coco_trt"
#MODEL_NAME = "faster_rcnn_nas_trt"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# The TensorRT inference graph file downloaded from Colab or your local machine.
trt_graph = get_frozen_graph(PATH_TO_FROZEN_GRAPH)

input_names = ['image_tensor']

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')


# Initiate video capture
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

    start_time = time.time()

    check, image = video.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image, axis=0)

    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image[None, ...]
    })


    # all outputs are float32 numpy arrays, so convert types as appropriate    boxes = boxes[0]  # index by 0 to remove batch dimension
    num_detections = int(num_detections[0])
    scores = scores[0]
    classes = classes[0].astype(np.uint8)
    boxes = boxes[0]

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index,
      #instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)


    fps = round(1 / (time.time() - start_time), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,"fps - " + str(fps),(0,50), font, 1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow("Object Detection.", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

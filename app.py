import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import re
import base64
import numpy as np
import os
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata
import tensorflow as tf
assert tf.__version__.startswith('2')
import platform
from typing import List, NamedTuple
import json


tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

fig = plt.figure()
#st.set_page_config(layout="wide")

def add_bg_from_url(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.title("Covid Compliance using AI")
st.markdown("Powered by [Think In Bytes](https://www.thinkinbytes.in)")

add_bg_from_url('white.jpg') 

st.sidebar.header("Behind the scenes !")
#st.markdown('<div style="text-align: justify;">Hello World!</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="text-align: justify;">This face-mask detection module is a demonstration of our light-weight AI enabled Computer Vision Engine that identifies pre-defined objects from the image. Our read-to-deploy pipeline features: </div>', unsafe_allow_html=True)
st.sidebar.markdown("")
st.sidebar.subheader("- Minimal Training")
st.sidebar.subheader("- Accurate Results")
st.sidebar.subheader("- Edge compatible")

Interpreter = tf.lite.Interpreter
load_delegate = tf.lite.experimental.load_delegate

# pylint: enable=g-import-not-at-top


class ObjectDetectorOptions(NamedTuple):

    enable_edgetpu = False
  
    label_allow_list = None
  
    label_deny_list = None
  
    max_results = -1
  
    num_threads = 1
  
    score_threshold = 0.0
    
class Rect(NamedTuple):
    left: float
    top: float
    right: float
    bottom: float


class Category(NamedTuple):
    label: str
    score: float
    index: int


class Detection(NamedTuple):
    """A detected object as the result of an ObjectDetector."""
    bounding_box: Rect
    categories: List[Category]


def edgetpu_lib_name():
    return {
      'Darwin': 'libedgetpu.1.dylib',
      'Linux': 'libedgetpu.so.1',
      'Windows': 'edgetpu.dll',
  }.get(platform.system(), None)


class ObjectDetector:
  
    _OUTPUT_LOCATION_NAME = 'location'
    _OUTPUT_CATEGORY_NAME = 'category'
    _OUTPUT_SCORE_NAME = 'score'
    _OUTPUT_NUMBER_NAME = 'number of detections'

    def __init__(self, model_path: str, options: ObjectDetectorOptions = ObjectDetectorOptions()) -> None:

        # Load metadata from model.
        displayer = metadata.MetadataDisplayer.with_model_file(model_path)

        # Save model metadata for preprocessing later.
        model_metadata = json.loads(displayer.get_metadata_json())
        process_units = model_metadata['subgraph_metadata'][0]['input_tensor_metadata'][0]['process_units']
        mean = 0.0
        std = 1.0
        for option in process_units:
            if option['options_type'] == 'NormalizationOptions':
                mean = option['options']['mean'][0]
                std = option['options']['std'][0]
        self._mean = mean
        self._std = std

        # Load label list from metadata.
        file_name = displayer.get_packed_associated_file_list()[0]
        label_map_file = displayer.get_associated_file_buffer(file_name).decode()
        label_list = list(filter(lambda x: len(x) > 0, label_map_file.splitlines()))
        self._label_list = label_list

        # Initialize TFLite model.
        if options.enable_edgetpu:
            if edgetpu_lib_name() is None:
                raise OSError("The current OS isn't supported by Coral EdgeTPU.")
                interpreter = Interpreter( model_path=model_path,
                  experimental_delegates=[load_delegate(edgetpu_lib_name())],
                  num_threads=options.num_threads)
        else:
            interpreter = Interpreter(model_path=model_path, num_threads=options.num_threads)

        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]

        # From TensorFlow 2.6, the order of the outputs become undefined.
        # Therefore we need to sort the tensor indices of TFLite outputs and to know
        # exactly the meaning of each output tensor. For example, if
        # output indices are [601, 599, 598, 600], tensor names and indices aligned
        # are:
        #   - location: 598
        #   - category: 599
        #   - score: 600
        #   - detection_count: 601
        # because of the op's ports of TFLITE_DETECTION_POST_PROCESS
        # (https://github.com/tensorflow/tensorflow/blob/a4fe268ea084e7d323133ed7b986e0ae259a2bc7/tensorflow/lite/kernels/detection_postprocess.cc#L47-L50).
        sorted_output_indices = sorted(
            [output['index'] for output in interpreter.get_output_details()])
        self._output_indices = {
            self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
            self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
            self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
            self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
        }

        self._input_size = input_detail['shape'][2], input_detail['shape'][1]
        self._is_quantized_input = input_detail['dtype'] == np.uint8
        self._interpreter = interpreter
        self._options = options

    def detect(self, input_image: np.ndarray) -> List[Detection]:

        image_height, image_width, _ = input_image.shape

        input_tensor = self._preprocess(input_image)

        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

        # Get all output details
        boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
        classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
        scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
        count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

        return self._postprocess(boxes, classes, scores, count, image_width,
                                 image_height)

    def _preprocess(self, input_image: np.ndarray) -> np.ndarray:
        """Preprocess the input image as required by the TFLite model."""

        # Resize the input
        
        #input_tensor = cv2.resize(input_image, self._input_size)
        resize_input = Image.fromarray(input_image)
        input_tensor = resize_input.resize((320,320))
        
        # Normalize the input if it's a float model (aka. not quantized)
        if not self._is_quantized_input:
            input_tensor = (np.float32(input_tensor) - self._mean) / self._std

        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor

    def _set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self._interpreter.get_input_details()[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, name):
        """Returns the output tensor at the given index."""
        output_index = self._output_indices[name]
        tensor = np.squeeze(self._interpreter.get_tensor(output_index))
        return tensor

    def _postprocess(self, boxes: np.ndarray, classes: np.ndarray,
                   scores: np.ndarray, count: int, image_width: int,
                   image_height: int) -> List[Detection]:

        results = []

        # Parse the model output into a list of Detection entities.
        for i in range(count):
            if scores[i] >= self._options.score_threshold:
                y_min, x_min, y_max, x_max = boxes[i]
                bounding_box = Rect(
                    top=int(y_min * image_height),
                    left=int(x_min * image_width),
                    bottom=int(y_max * image_height),
                    right=int(x_max * image_width))
                class_id = int(classes[i])
                category = Category(
                    score=scores[i],
                    label=self._label_list[class_id],  # 0 is reserved for background
                    index=class_id)
                result = Detection(bounding_box=bounding_box, categories=[category])
                results.append(result)

        # Sort detection results by score ascending
        sorted_results = sorted(
            results,
            key=lambda detection: detection.categories[0].score,
            reverse=True)

        # Filter out detections in deny list
        filtered_results = sorted_results
        if self._options.label_deny_list is not None:
            filtered_results = list(
              filter(
                  lambda detection: detection.categories[0].label not in self.
                  _options.label_deny_list, filtered_results))

        # Keep only detections in allow list
        if self._options.label_allow_list is not None:
            filtered_results = list(
              filter(
                  lambda detection: detection.categories[0].label in self._options.
                  label_allow_list, filtered_results))

        # Only return maximum of max_results detection.
        if self._options.max_results > 0:
            result_count = min(len(filtered_results), self._options.max_results)
            filtered_results = filtered_results[:result_count]

        return filtered_results


_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


def visualize(image: np.ndarray,detections: List[Detection],) -> np.ndarray:

    for detection in detections:
        # Draw bounding_box
        if detection.categories[0].score > 0.5:
            start_point = detection.bounding_box.left, detection.bounding_box.top
            end_point = detection.bounding_box.right, detection.bounding_box.bottom
            #cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.rectangle((start_point[0],start_point[1],end_point[0],end_point[1]), fill=None)
            
            
            
            # Draw label and score
            category = detection.categories[0]
            class_name = category.label
            probability = round(category.score, 2)
            result_text = class_name + ' (' + str(probability) + ')'
            text_location = (_MARGIN + detection.bounding_box.left,
                             _MARGIN + _ROW_SIZE + detection.bounding_box.top)
            draw.text((text_location),result_text,(255,255,255))
            image = np.array(image)
            #cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
            #           _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    return image, class_name




def main():
    DETECTION_THRESHOLD = 0.5 #@param {type:"number"}
    TFLITE_MODEL_PATH = "android.tflite" #@param {type:"string"}

    #file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    with st.container():
            file_uploaded = st.camera_input("Take a picture")
            if file_uploaded is not None:    
                image = Image.open(file_uploaded)
                image.thumbnail((512, 512), Image.ANTIALIAS)
                image_np = np.asarray(image)
                options = ObjectDetectorOptions()
                detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)

                # Run object detection estimation using the model.
                detections = detector.detect(image_np)

                # Draw keypoints and edges on input image
                image_np, class_name = visualize(image_np, detections)

                plt.imshow(image_np)
                plt.figure(figsize = (1,1.5))
                plt.axis("off")
                st.pyplot(fig)
                st.text("Tip: Try to cover your mouth with hand and see if you can fool AI !")

                
if __name__ == "__main__":
    main()
    with st.container():
        st.markdown("<h2 style='text-align: center; color: black;'>Object Detection - Applications</h2>", unsafe_allow_html=True)
        image = Image.open('screen3.png')
        st.image(image)


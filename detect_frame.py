from keras_yolo3.yolo3.utils import letterbox_image
import numpy as np
from keras import backend as K


def detect_frame(yolo, image, learning_phase=0):
    if yolo.model_image_size != (None, None):
        assert yolo.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert yolo.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(yolo.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = yolo.sess.run(
        [yolo.boxes, yolo.scores, yolo.classes],
        feed_dict={
            yolo.yolo_model.input: image_data,
            yolo.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): learning_phase
        })
    return out_boxes, out_scores, out_classes

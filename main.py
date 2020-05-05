import logging
import tensorflow as tf
import os
import dtlpy as dl
from keras_yolo3.yolo import YOLO
from deep_sort.tools import generate_detections as gdet
from detect_video import detect_video

logger = logging.getLogger(name=__name__)


class ServiceRunner():
    """
    Package runner class

    """

    def __init__(self, package_name, **kwargs):
        """
        Init package attributes here

        :param kwargs: config params
        :return:
        """
        if not os.path.isdir('model_data'):
            os.makedirs('model_data')
            # download artifacts
        package = dl.packages.get(package_name=package_name)

        if not os.path.isfile('model_data/yolo.h5'):
            artifact = package.project.artifacts.get(package_name=package_name,
                                                     artifact_name='yolo.h5')
            artifact.download(local_path='model_data')
        if not os.path.isfile('model_data/yolo_anchors.txt'):
            artifact = package.project.artifacts.get(package_name=package_name,
                                                     artifact_name='yolo_anchors.txt')
            artifact.download(local_path='model_data')
        if not os.path.isfile('model_data/coco_classes.txt'):
            artifact = package.project.artifacts.get(package_name=package_name,
                                                     artifact_name='coco_classes.txt')
            artifact.download(local_path='model_data')
        if not os.path.isfile('model_data/mars-small128.pb'):
            artifact = package.project.artifacts.get(package_name=package_name,
                                                     artifact_name='mars-small128.pb')
            artifact.download(local_path='model_data')

        ###############
        # load models #
        ###############
        self.yolo = YOLO()
        self.encoder = gdet.create_box_encoder('model_data/mars-small128.pb', batch_size=1)
        self.graph = tf.get_default_graph()

    def run(self, item, progress=None):
        assert isinstance(item, dl.Item)
        logger.info('Downloading video')
        video_filepath = item.download()
        try:
            logger.info('Running dection on video')
            annotations_dict = detect_video(graph=self.graph,
                                            yolo=self.yolo,
                                            encoder=self.encoder,
                                            video_filepath=video_filepath,
                                            mark_on_video=False,
                                            show=False)

            logger.info('Building Dataloop annotations')
            annotations_builder = item.annotations.builder()
            for i_frame, annotations in annotations_dict.items():
                for annotation in annotations:
                    annotations_builder.add(annotation_definition=dl.Box(top=annotation['top'],
                                                                         left=annotation['left'],
                                                                         bottom=annotation['bottom'],
                                                                         right=annotation['right'],
                                                                         label=annotation['label']),
                                            object_id=annotation['object_id'],
                                            frame_num=i_frame,
                                            end_frame_num=i_frame,
                                            )
            logger.info('Uploading annotations...')
            item.annotations.upload(annotations_builder)
            logger.info('Done! detection and annotations finished successfully')
        finally:
            if os.path.isfile(video_filepath):
                os.remove(video_filepath)


if __name__ == "__main__":
    """
    Run this main to locally debug your package
    """
    # dl.packages.test_local_package()

import cv2
import numpy as np
from PIL import Image
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from detect_frame import detect_frame


def detect_video(graph, yolo, encoder, video_filepath, mark_on_video, show):
    global_object_ids = dict()
    object_id_counter = 0

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 0.3  # 1.0

    # read video filepath
    cap = cv2.VideoCapture(video_filepath)

    try:
        # per class object tracker
        class_tracker_dict = dict()

        i_frame = -1
        detection_dict = {}
        while True:
            ret, frame = cap.read()
            i_frame += 1
            if not ret:
                break
            with graph.as_default():
                out_boxes, out_scores, out_classes = detect_frame(yolo=yolo,
                                                                  image=Image.fromarray(frame))
            # convert to [x,y,w,h]
            boxs = np.array([[bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0]] for bb in out_boxes])

            # add detections to class tracker dict
            for class_id in np.unique(out_classes):
                if class_id not in class_tracker_dict:
                    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
                    class_tracker_dict[class_id] = Tracker(metric)

            # update all trackers with incoming detections
            for class_id, tracker in class_tracker_dict.items():
                inds = out_classes == class_id
                with graph.as_default():
                    features = encoder(frame, boxs[inds])
                detections = [Detection(bbox, score, feature) for
                              bbox, score, feature in zip(boxs[inds], out_scores[inds], features)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                class_tracker_dict[class_id].predict()
                class_tracker_dict[class_id].update(detections)

            # Save detection per frame per object_id
            detection_dict[i_frame] = list()
            for class_id, tracker in class_tracker_dict.items():
                for track in tracker.tracks:
                    # if not track.is_confirmed() or track.time_since_update > 1:
                    if track.time_since_update > 2:
                        continue
                    bbox = track.to_tlbr()
                    object_name = '{}_{}'.format(yolo.class_names[class_id], track.track_id)
                    if object_name not in global_object_ids:
                        global_object_ids[object_name] = object_id_counter
                        object_id_counter += 1

                    # mark in video
                    label = yolo.class_names[class_id]
                    left = int(bbox[0])
                    top = int(bbox[1])
                    right = int(bbox[2])
                    bottom = int(bbox[3])
                    object_id = track.track_id

                    if mark_on_video:
                        cv2.rectangle(frame, (left, top), (right, bottom),
                                      (255, 255, 255), 2)
                        cv2.putText(frame, '{}_{}'.format(label, object_id),
                                    (int(bbox[0]), int(bbox[1])),
                                    0, 1, (0, 255, 0), 1)
                        # create results dictionary

                    detection_dict[i_frame].append({'top': top,
                                                    'left': left,
                                                    'right': right,
                                                    'bottom': bottom,
                                                    'label': label,
                                                    'object_id': global_object_ids[object_name]})
            if show:
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("result", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
    return detection_dict


if __name__ == "__main__":
    import sys
    import tensorflow as tf
    from keras_yolo3.yolo import YOLO
    from deep_sort.tools import generate_detections as gdet

    # load model and tracker encoder
    yolo = YOLO()
    encoder = gdet.create_box_encoder('model_data/mars-small128.pb', batch_size=1)
    graph = tf.get_default_graph()
    # run on video
    video_filepath = sys.argv[0]
    detect_video(graph=graph,
                 yolo=yolo,
                 encoder=encoder,
                 video_filepath=video_filepath,
                 mark_on_video=True,
                 show=True)

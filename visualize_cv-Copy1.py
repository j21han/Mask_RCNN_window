from __future__ import print_function
import cv2
import numpy as np
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils

from threading import Thread, Lock



def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


if __name__ == '__main__':
    """
        test everything
    """
    import os
    import sys

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    
    #ROOT_DIR = os.getcwd()
    ROOT_DIR = os.path.abspath("./") # 최상위 경로로 지정
    sys.path.append(ROOT_DIR)
    
    from mrcnn import utils
    import mrcnn.model as modellib
    print(ROOT_DIR)    
    sys.path.append(ROOT_DIR+"\\samples\\coco\\")
    print(os.path.join(ROOT_DIR,"\\samples\\coco\\"))
    import coco
    
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU =1

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = [
        'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    class WebcamVideoStream :
        def __init__(self, src = 1, width = 800, height = 800) :
            self.stream = cv2.VideoCapture(src+cv2.CAP_DSHOW)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            (self.grabbed, self.frame) = self.stream.read()
            self.started = False
            self.read_lock = Lock()
            

        
        def start(self) :
            if self.started:
                print("already started!!")
                return None
            self.started = True
            self.thread = Thread(target=self.update, args=())
            self.thread.start()
            return self

        def read(self) :
            self.read_lock.acquire()
            frame = self.frame.copy()
            self.read_lock.release()
            return frame

        def update(self) :
            while self.started :
                (grabbed, frame) = self.stream.read()
                self.read_lock.acquire()
                self.grabbed, self.frame = grabbed, frame
                self.read_lock.release()

        def stop(self) :
            self.started = False
            self.thread.join()

        def __exit__(self, exc_type, exc_value, traceback) :
            self.stream.release()

        
if __name__ == "__main__" :
    vs = WebcamVideoStream().start()
    while True :
        frame = vs.read()
        results = model.detect([frame], verbose=0)
        
        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27 :
            break

    vs.release()
    cv2.destroyAllWindows()

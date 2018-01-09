import _init_paths_frcnn
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
import numpy as np
import caffe, os, sys, cv2
from utils.timer import Timer
import argparse


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def vis_detections(im, class_name, dets, fw, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0:
        return

    fw.write(class_name + ":")
    cnt = len(inds)

    im = im[:, :, (2, 1, 0)]
    for i in inds:
        cnt -= 1
        bbox = dets[i, :4]
        # score = dets[i, -1] # You could get the score if needed
        # Get the bounding box of the detected object
        x, y, width, height = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        if cnt > 0:
            fw.write('{} {} {} {} '.format(x, y, width, height))
        else:
            fw.write('{} {} {} {}\n'.format(x, y, width, height))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN detector')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='faster_rcnn_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def detect(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the image
    im_file = os.path.join(cfg.DATA_DIR, 'images', image_name)
    im = cv2.imread(im_file)

    # Open the file to write detected information
    im_name = os.path.splitext(image_name)[0] + '.bbox'
    detect_im_file = os.path.join(cfg.DATA_DIR, 'result', 'bbox', im_name)
    fw = open(detect_im_file, 'w')

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, fw, thresh=CONF_THRESH)

    # End writing file
    fw.close()

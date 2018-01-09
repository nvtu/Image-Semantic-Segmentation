import _init_paths_seg
from fast_rcnn.config import cfg
import os, caffe, math, argparse, glob, cv2
import numpy as np, bbox_utils
from PIL import Image


COLORS = {'__background__' : (0, 0, 0),
           'aeroplane' : (255, 255, 255), 'bicycle' : (255, 0, 0), 'bird' : (128, 0, 0), 'boat' : (255, 255, 0),
           'bottle' : (128, 128, 0), 'bus' : (0, 255, 0), 'car' : (0, 128, 0), 'cat' : (0, 255, 255), 'chair' : (0, 128, 128),
           'cow' : (0, 0, 255), 'diningtable' : (0, 0, 128), 'dog' : (255, 160, 122), 'horse' : (128, 0, 128),
           'motorbike' : (49, 125, 237), 'person' : (142, 208, 169), 'pottedplant' : (91,91,255),
           'sheep' : (148, 218, 231), 'sofa' : (144, 140, 210), 'train' : (255, 87, 51), 'tvmonitor' : (252, 121, 71)}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Fully Convolutional Networks')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args


def load_detected_bbox(image_name):
    im_name = os.path.splitext(image_name)[0] + '.bbox'
    bbox_path = os.path.join(cfg.DATA_DIR, 'result', 'bbox', im_name)
    fr = open(bbox_path, 'r')
    content = fr.read()
    fr.close()
    categories = content.split('\n')
    cls_name = []
    bbox = []
    for ind in xrange(len(categories) - 1):
        container = categories[ind].split(':')
        key = container[0]
        vals = container[1].split(' ')
        cls_name.append(key)
        cls_bbox = []
        bbox_info = []
        for i in xrange(len(vals)):
            bbox_info.append(int(math.ceil(float(vals[i]))))
            if i % 4 == 3:
                cls_bbox.append(bbox_info)
                bbox_info = []
        bbox.append(cls_bbox)
    return cls_name, bbox

# def color_region(im, s_mask, cls_name, x, y, threshold = 75):
#     color = COLORS[cls_name]
#     width, height = s_mask.shape[1], s_mask.shape[0]
#     for i in xrange(width):
#         for j in xrange(height):
#             if s_mask[j, i] >= threshold and im[y + j, x + i, :].all() == 0:
#                 im[y + j, x + i, :] = color
#     return im

# def refinement_color(im, s_mask, cls_name, x, y, width, height, threshold = 254):
#     color = COLORS[cls_name]
#     for i in xrange(width):
#         for j in xrange(height):
#             if s_mask[y + j, x + i] >= threshold and im[y + j, x + i, :].all() == 0:
#                 im[y + j, x + i, :] = color
#     return im

def segment():
    args = parse_args()
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    # prototxt = os.path.join(cfg.MODELS_DIR, 'DHSNet_model', 'SO_RCL_deploy.prototxt')
    # caffemodel = os.path.join(cfg.DATA_DIR, 'DHSNet_data', 'SO_RCL_models_iter_10000.caffemodel')
    net = caffe.Net('./data/voc-fcn8s/deploy.prototxt', './data/voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)

    # print '\n\nLoaded network {:s}'.format(caffemodel)
    im_path = os.path.join(cfg.DATA_DIR, 'images')
    im_names = glob.glob(im_path + '/*.jpg')
    for im_path in im_names:
        im_name = os.path.basename(im_path)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Load data/images/{}'.format(im_name)
        image = cv2.imread(im_path)
        result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        print 'Load detected bounding box of data/images/{}'.format(im_name)
        cls_names, bbox = load_detected_bbox(im_name)
        bboxTool = bbox_utils.bboxUtils(cls_names, bbox)
        print 'Semantic segmentation for data/images/{}'.format(im_name)
        # Get mask
        im = Image.open(im_path)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        mask = net.blobs['score'].data[0].argmax(axis=0)
        # Refinement saliency mask
        # sm_full = predict_SOs.predict(net, image)
        
        for ind in xrange(len(cls_names)):
            cls_name = cls_names[ind]
            cls_bbox = bbox[ind]
            for j in xrange(len(cls_bbox)):
                bbox_info = cls_bbox[j]
                x, y, width, height = bbox_info[0], bbox_info[1], bbox_info[2], bbox_info[3]
                sub_mask = mask[y : y + height, x : x + width] 
                im = image[y : y + height, x : x + width, :]
                crop_path = os.path.join(cfg.DATA_DIR, 'result', 'detected_objects', '{}_{}_{}.jpg'.format(im_name, cls_name, j))
                cv2.imwrite(crop_path, im)
                result_image = bboxTool.colorCls_region(result_image, sub_mask, bbox_info, cls_name, COLORS[cls_name])
                # sm = predict_SOs.predict(net, im)
                # # Save result of each step to files
                # sal_path = os.path.join(cfg.DATA_DIR, 'tmp', 'saliency_maps', '{}_{}_{}.jpg'.format(im_name, cls_name, j))
                # cv2.imwrite(sal_path, sub_mask)
                # result_image = color_region(result_image, sm, cls_name, x, y)
                # result_image = refinement_color(result_image, sm_full, cls_name, x, y, width, height)
        result_path = os.path.join(cfg.DATA_DIR, 'result', 'segment_result', im_name)
        # full_sal_path = os.path.join(cfg.DATA_DIR, 'result', 'saliency_maps', im_name)
        cv2.imwrite(result_path, result_image)
        # cv2.imwrite(full_sal_path, sm_full)
    del net

if __name__ == '__main__':
    segment()
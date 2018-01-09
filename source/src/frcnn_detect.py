import _init_paths_frcnn
import faster_rcnn_detect as frcnn
from fast_rcnn.config import cfg
import caffe, os, sys, glob, math

def detect_object():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = frcnn.parse_args()
    prototxt = os.path.join(cfg.MODELS_DIR, frcnn.NETS[args.faster_rcnn_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              frcnn.NETS[args.faster_rcnn_net][1])
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    im_path = os.path.join(cfg.DATA_DIR, 'images')
    im_names = glob.glob(im_path + '/*.jpg')
    for im_path in im_names:
        im_name = os.path.basename(im_path)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Detecting data/images/{}'.format(im_name)
        frcnn.detect(net, im_name)
    del net

if __name__ == '__main__':
    detect_object()

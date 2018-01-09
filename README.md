# Image Semantic Segmentation on PASCAL_VOC_0712 

**Authors**: Tu-Khiem Le, Van-Tu Ninh

## Our Contribution

  - Propose solution to effectively segmentation overlapping objects of many classes in an image
where fully convolutional networks used in semantic segmentation fails to properly generates
correct class segmentation, especially on overlapping objects in an image.
  - Propose semantic segmentation preprocessing of object detection using Faster RCNN to corroborate classes segmentation result.
  - Propose to use fully convolutional networks result as mask to color the objects in image.
  - Propose solution to color overlapping regions of objects in image.

## Comparison between our proposed method and outputs of Fully Convolutional Networks.
![Comparison](https://github.com/nvtu/Image-Semantic-Segmentation/blob/master/result.jpeg?raw=true)
- The first row in the image is the original image used as input in our segmentation system. 
- The second row is the output of semantic segmentation using fully convolutional networks. 
- The third row is the output of our proposed system combining the strength of both Faster RCNN and Fully Convolutional Networks with our proposed region coloring method for segmentation system. 
## Installation
#### Source folder configuration
- **data**: contains the data of Faster RCNN models saved in subfolder faster_rcnn_models, Fully Convolutional Networks models saved in subfolder voc-fcn8s, testing images saved in subfolder images, execution result saved in subfolder result.
- **src**: contains the source code of our project
- **models**: contains the architecture of Faster RCNN to load in running process.
- **caffe**: caffe framework for Fully Convolutional Networks code.
- **caffe-fast-rcnn**: caffe framework for Fast RCNN code.
- **lib**: extra library for Faster RCNN code.
#### Caffe deployment
Download caffe library from two repositories of Faster RCNN and Fully Convolutional Networks respectively as below:
- https://github.com/rbgirshick/py-faster-rcnn.
- https://github.com/shelhamer/fcn.berkeleyvision.org.`

Following the instruction from the main site of caffe: http://caffe.berkeleyvision.org/installation.html.
**Notice**: If you use build caffe with CUDA and CUDNN enabled, you should use CUDNN version 4.0.

#### Run our code
1. Download pre-trained model into folder /data/faster_rcnn_models by executing file **faster_rcnn_models/fetch_faster_rcnn_models.sh**.
2.  Download pre-trained model into folder /data/voc-fcn8s by following link in file **caffemodel-url**.
3.  Put images in data/images folder and clean the result in data/result/bbox, data/result/detected_objects, data/result/segment_result
4.  Execute frcnn_detect.py by running ```python src/frcnn_detect.py```
5.  Execute segment.py by running ```python src/segment.py```

The result is saved in /data/result/segment_result. You can see the result of each process in each
folder in /data/result.

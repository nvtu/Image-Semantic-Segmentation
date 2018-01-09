import operator, cv2
class bboxUtils(object):

    def __init__(self, cls_name, bbox):
        self.cls_names = cls_name
        self.bboxes = bbox
    
    def set_data(self, cls_name, bbox):
        self.cls_names = cls_name
        self.bboxes = bbox
    
    def intersection_regions(self, det_bbox, cls):
        intersection = []
        dx, dy, dwidth, dheight = det_bbox[0], det_bbox[1], det_bbox[2], det_bbox[3]
        dxmax, dymax = dx + dwidth, dy + dheight
        for i in xrange(len(self.cls_names)):
            cls_name = self.cls_names[i]
            if cls_name != cls:
                bbox = self.bboxes[i]
                for j in xrange(len(bbox)):
                    x, y, width, height = bbox[j][0], bbox[j][1], bbox[j][2], bbox[j][3]
                    xmax, ymax = x + width, y + height
                    ans_x, ans_y = max(x, dx), max(y, dy)
                    ans_xmax, ans_ymax = min(dxmax, xmax), min(dymax, ymax)
                    if ans_x >= dx and ans_y >= dy and ans_xmax <= dxmax and ans_ymax <= dymax and ans_xmax > ans_x and ans_ymax > ans_y:
                        intersection.append([ans_x, ans_y, ans_xmax - ans_x, ans_ymax - ans_y])
        return intersection 

    def is_in_intersection(self, x, y, reg):
        rx, ry, rxmax, rymax = reg[0], reg[1], reg[0] + reg[2], reg[1] + reg[3]
        if x >= rx and x <= rxmax and y >= ry and y <= rymax:
            return True
        return False

    def colorCls_region(self, im, mask, det_bbox, cls, color):
        intersection = self.intersection_regions(det_bbox, cls)
        x, y, width, height = det_bbox[0], det_bbox[1], det_bbox[2], det_bbox[3]
        cnt = {}
        for i in xrange(width):
            for j in xrange(height):
                if im[y + j, x + i, :].all() == 0:
                    is_in = False
                    for reg in intersection:
                        if self.is_in_intersection(x + i, y + j, reg):
                            is_in = True
                            break
                    if is_in == False:
                        try:
                            val = mask[j, i]
                            cnt[val] += 1
                        except KeyError:
                            cnt[val] = 1
        cnt[0] = 0
        ind_max = max(cnt.iteritems(), key=operator.itemgetter(1))[0]
        if ind_max > 0:
            for i in xrange(width):
                for j in xrange(height):
                    if im[y + j, x + i, :].all() == 0 and mask[j, i] == ind_max:
                        im[y + j, x + i, :] = color
        return im
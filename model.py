import numpy as np
import tensorflow as tf
from helper_api import HelperAPI
import cv2


class YOLO:
    def __init__(self):
        self.weights_file = 'YOLO_small.ckpt'
        self.threshold = 0.1
        self.iou_threshold = 0.3
        self.num_class = 20
        self.num_box = 2
        self.grid_size = 7
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.w_img = 448
        self.h_img = 448
        # training variaible
        self.training = False
        self.keep_prob = tf.placeholder(tf.float32)
        self.lambdacoord = 5.0
        self.lambdanoobj = 0.5
        self.label = None
        self.label = None
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def build_model(self, input):
        # self.x = tf.placeholder('float32', [None, 448, 448, 3])
        self.conv_1 = HelperAPI.conv_layer(1, input, 64, 7, 2)
        self.pool_2 = HelperAPI.pooling_layer(2, self.conv_1, 2, 2)

        self.conv_3 = HelperAPI.conv_layer(3, self.pool_2, 192, 3, 1)
        self.pool_4 = HelperAPI.pooling_layer(4, self.conv_3, 2, 2)

        self.conv_5 = HelperAPI.conv_layer(5, self.pool_4, 128, 1, 1)
        self.conv_6 = HelperAPI.conv_layer(6, self.conv_5, 256, 3, 1)
        self.conv_7 = HelperAPI.conv_layer(7, self.conv_6, 256, 1, 1)
        self.conv_8 = HelperAPI.conv_layer(8, self.conv_7, 512, 3, 1)
        self.pool_9 = HelperAPI.pooling_layer(9, self.conv_8, 2, 2)

        self.conv_10 = HelperAPI.conv_layer(10, self.pool_9, 256, 1, 1)
        self.conv_11 = HelperAPI.conv_layer(11, self.conv_10, 512, 3, 1)
        self.conv_12 = HelperAPI.conv_layer(12, self.conv_11, 256, 1, 1)
        self.conv_13 = HelperAPI.conv_layer(13, self.conv_12, 512, 3, 1)
        self.conv_14 = HelperAPI.conv_layer(14, self.conv_13, 256, 1, 1)
        self.conv_15 = HelperAPI.conv_layer(15, self.conv_14, 512, 3, 1)
        self.conv_16 = HelperAPI.conv_layer(16, self.conv_15, 256, 1, 1)
        self.conv_17 = HelperAPI.conv_layer(17, self.conv_16, 512, 3, 1)
        self.conv_18 = HelperAPI.conv_layer(18, self.conv_17, 512, 1, 1)
        self.conv_19 = HelperAPI.conv_layer(19, self.conv_18, 1024, 3, 1)
        self.pool_20 = HelperAPI.pooling_layer(20, self.conv_19, 2, 2)

        self.conv_21 = HelperAPI.conv_layer(21, self.pool_20, 512, 1, 1)
        self.conv_22 = HelperAPI.conv_layer(22, self.conv_21, 1024, 3, 1)
        self.conv_23 = HelperAPI.conv_layer(23, self.conv_22, 512, 1, 1)
        self.conv_24 = HelperAPI.conv_layer(24, self.conv_23, 1024, 3, 1)
        self.conv_25 = HelperAPI.conv_layer(25, self.conv_24, 1024, 3, 1, trainable=self.training)
        self.conv_26 = HelperAPI.conv_layer(26, self.conv_25, 1024, 3, 2, trainable=self.training)
        self.conv_27 = HelperAPI.conv_layer(27, self.conv_26, 1024, 3, 1, trainable=self.training)
        self.conv_28 = HelperAPI.conv_layer(28, self.conv_27, 1024, 3, 1, trainable=self.training)

        self.fc_29 = HelperAPI.fc_layer(29, self.conv_28, 512, flat=True, linear=False, trainable=self.training)
        self.fc_30 = HelperAPI.fc_layer(30, self.fc_29, 4096, flat=False, linear=False, trainable=self.training)
        # self.drop_31 = HelperAPI.dropout(31, tf.cast(self.fc_30,dtype=tf.float64))
        self.fc_32 = HelperAPI.fc_layer(32, self.fc_30, 1470, flat=False, linear=True, trainable=self.training)

    def process_output(self, output):
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= self.w_img
        boxes[:, :, :, 1] *= self.h_img
        boxes[:, :, :, 2] *= self.w_img
        boxes[:, :, :, 3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0: continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.clac_iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def clac_iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2],
                                                                         box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3],
                                                                         box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def show_result(self, img, results):
        img_cp = img.copy()
        result = []
        labels = []
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2
            result.append([x,y,h,w])
            labels.append(results[i][0])
            cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite('out_images.jpg', img_cp)
        return labels,result

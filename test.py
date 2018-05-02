import numpy as np
import tensorflow as tf
import cv2
from model import YOLO

weights_file = 'YOLO_small.ckpt'

def predict(img):
    print("######################")
    img_resized = cv2.resize(img, (448, 448))
    img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized_np = np.asarray(img_RGB)
    inputs = np.zeros((1, 448, 448, 3), dtype='float32')
    inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
    with tf.Graph().as_default():
        input = tf.placeholder('float32', [None, 448, 448, 3])
        model = YOLO()
        model.build_model(input=input)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, weights_file)
            in_dict = {input: inputs, model.keep_prob: 1.0}
            net_output = sess.run(model.fc_32, feed_dict=in_dict)
            result = model.process_output(net_output[0])
            labels ,results = model.show_result(img=img_resized, results=result)

    return labels ,results


if __name__ == "__main__":
    img = cv2.imread('images.jpg')
    predict(img=img)

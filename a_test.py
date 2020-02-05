
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf 
import time
import a_forward
import a_backward
import numpy as np
import a_generateds

TEST_INTERVAL_SECS = 200
TEST_NUM = 2000      #此处手动输入训练集样本数

def test():#为要喂入的数据集
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,[
        TEST_NUM,
        a_forward.BOARD_SIZE, 
        a_forward.BOARD_SIZE,
        a_forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32,[None, a_forward.OUTPUT_NODE])
        y = a_forward.forward(x, False, None)

        #实例化可还原滑动平均值的saver
        ema = tf.train.ExponentialMovingAverage(a_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        img_batch, label_batch = a_generateds.test_input_pipeline(TEST_NUM)

        #计算正确率
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        #correct_prediction = (np.array((y - y_)) < 0.1).all() == True
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(a_backward.MODEL_SAVE_PATH)#加载ckpt模型
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    xs, ys = sess.run([img_batch, label_batch]) 
                    reshaped_xs = np.reshape(xs,(
                    TEST_NUM,
                    a_forward.BOARD_SIZE,
                    a_forward.BOARD_SIZE,
                    a_forward.NUM_CHANNELS))
                    accuracy_score = sess.run(accuracy,feed_dict={x:reshaped_xs,y_:ys})
                    print("After %s training steps,test accuracy = %g"%(global_step, accuracy_score))

                    coord.request_stop()
                    coord.join(threads)

                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    test()

if __name__ == '__main__':
    main()


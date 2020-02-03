
import tensorflow as tf
import a_forward
import os
import numpy as np
import a_generateds
import time

BATCH_SIZE = 300
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 1e-8
STEPS = 2000000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "a_model"
train_num_examples = 145242        #训练总样本数需要手动确定


#定义反向传播过程
def backward():#喂入神经网络的数据集

    time_start = time.time()
    
    x = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        a_forward.BOARD_SIZE, 
        a_forward.BOARD_SIZE,
        a_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32,[None,a_forward.OUTPUT_NODE])
    y = a_forward.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)#轮数计数器
    
    #损失函数loss含正则化regularization
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses')) 
    
    #指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    
    #定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #定义滑动平均值
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    
    #实例化saver
    saver = tf.train.Saver()
    chessboard_batch, label_batch = a_generateds.input_pipeline(BATCH_SIZE,num_epochs=None)
    
    sess = tf.Session()
    init_op = [tf.global_variables_initializer(), tf.global_variables_initializer()]#初始化
    sess.run(init_op)
        
    #断点续训
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(STEPS):
        xs, ys = sess.run([chessboard_batch, label_batch])
        reshaped_xs = np.reshape(xs,(
            BATCH_SIZE,
            a_forward.BOARD_SIZE,
            a_forward.BOARD_SIZE,
            a_forward.NUM_CHANNELS
        ))
        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs,y_: ys})
            
        if i % 1000 == 0:
            time_end = time.time()
            print("After %d training steps,loss on training batch is %g,totally cost %d s"%(step,loss_value,time_end-time_start))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)

    coord.request_stop()
    coord.join(threads)

def main():
    backward()

if __name__ == '__main__':
    main()
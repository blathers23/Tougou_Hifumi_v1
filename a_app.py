
import tensorflow as tf 
import numpy as np 
import a_backward
import a_forward

def restore_model(testBoard):
    board_ready = np.reshape(testBoard,(
                    1,
                    a_forward.BOARD_SIZE,
                    a_forward.BOARD_SIZE,
                    a_forward.NUM_CHANNELS))
    #print(board_ready)
    
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32,[
        1,
        a_forward.BOARD_SIZE, 
        a_forward.BOARD_SIZE,
        a_forward.NUM_CHANNELS])#重现计算图
        y = a_forward.forward(x,False,None)#计算求得y
        preValue = tf.argmax(y,1)#y的最大值对应的列表索引号即为最大值

        variable_averages = tf.train.ExponentialMovingAverage(a_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)#实例化带有滑动平均值的saver

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(a_backward.MODEL_SAVE_PATH)#用with结构加载ckpt
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x:board_ready})
                #y = sess.run(y, feed_dict={x:board_ready})
                #print(y)
                return preValue
            else:
                print("No checkpoint file found")
                return -1

def application():
    test = input("please input the test board:")
    testBoard=[float(n) for n in test.split(',')]
    print(testBoard)
    preValue = restore_model(testBoard)
    print("The prediction move is:%d"%(preValue))

def main():
    application()

if __name__ == '__main__':
    main()
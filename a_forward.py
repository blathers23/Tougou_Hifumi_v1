
import tensorflow as tf 

INPUT_NODE = 25
BOARD_SIZE = 5
NUM_CHANNELS = 1
CONV1_SIZE = 2
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 2
CONV2_KERNEL_NUM = 64
CONV3_SIZE = 2
CONV3_KERNEL_NUM = 128
CONV4_SIZE = 2
CONV4_KERNEL_NUM = 256
CONV5_SIZE = 2
CONV5_KERNEL_NUM = 256
CONV6_SIZE = 2
CONV6_KERNEL_NUM = 256
CONV7_SIZE = 2
CONV7_KERNEL_NUM = 256
CONV8_SIZE = 2
CONV8_KERNEL_NUM = 512
FC1_SIZE = 1024
FC2_SIZE = 2048
FC3_SIZE = 1024
OUTPUT_NODE = 3

#定义权重
def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    #损失函数loss含正则化regularization
    if regularizer != None: tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

#定义偏差值
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

#卷积计算函数
def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides = [1,1,1,1],padding = 'SAME') 

#最大池化生成函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') 

#定义前向传播
def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM],regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM , CONV2_KERNEL_NUM],regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    conv3_w = get_weight([CONV3_SIZE, CONV3_SIZE, CONV2_KERNEL_NUM , CONV3_KERNEL_NUM],regularizer)
    conv3_b = get_bias([CONV3_KERNEL_NUM])
    conv3 = conv2d(pool2, conv3_w)
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))
    pool3 = max_pool_2x2(relu3)

    conv4_w = get_weight([CONV4_SIZE, CONV4_SIZE, CONV3_KERNEL_NUM , CONV4_KERNEL_NUM],regularizer)
    conv4_b = get_bias([CONV4_KERNEL_NUM])
    conv4 = conv2d(pool3, conv4_w)
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_b))
    pool4 = max_pool_2x2(relu4)

    conv5_w = get_weight([CONV5_SIZE, CONV5_SIZE, CONV4_KERNEL_NUM , CONV5_KERNEL_NUM],regularizer)
    conv5_b = get_bias([CONV5_KERNEL_NUM])
    conv5 = conv2d(pool4, conv5_w)
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_b))
    pool5 = max_pool_2x2(relu5)
    '''
    conv6_w = get_weight([CONV6_SIZE, CONV6_SIZE, CONV5_KERNEL_NUM , CONV6_KERNEL_NUM],regularizer)
    conv6_b = get_bias([CONV6_KERNEL_NUM])
    conv6 = conv2d(pool5, conv6_w)
    relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_b))
    pool6 = max_pool_2x2(relu6)
    
    conv7_w = get_weight([CONV7_SIZE, CONV7_SIZE, CONV6_KERNEL_NUM , CONV7_KERNEL_NUM],regularizer)
    conv7_b = get_bias([CONV7_KERNEL_NUM])
    conv7 = conv2d(pool6, conv7_w)
    relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_b))
    pool7 = max_pool_2x2(relu7)
    
    conv8_w = get_weight([CONV8_SIZE, CONV8_SIZE, CONV7_KERNEL_NUM , CONV8_KERNEL_NUM],regularizer)
    conv8_b = get_bias([CONV8_KERNEL_NUM])
    conv8 = conv2d(pool7, conv8_w)
    relu8 = tf.nn.relu(tf.nn.bias_add(conv8, conv8_b))
    pool8 = max_pool_2x2(relu8)
    '''

    pool_shape = pool5.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] #三维数组的长度，宽度，深度
    reshaped = tf.reshape(pool5, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes,FC1_SIZE],regularizer)
    fc1_b = get_bias([FC1_SIZE])
    fc1 = tf.nn.tanh(tf.matmul(reshaped ,fc1_w) + fc1_b)
    if train: fc1 = tf.nn.dropout(fc1, 0.5)
    
    fc2_w = get_weight([FC1_SIZE,FC2_SIZE],regularizer)
    fc2_b = get_bias([FC2_SIZE])
    fc2 = tf.nn.relu(tf.matmul(fc1 ,fc2_w) + fc2_b)
    if train: fc2 = tf.nn.dropout(fc2, 0.5)
    '''
    fc3_w = get_weight([FC2_SIZE,FC3_SIZE],regularizer)
    fc3_b = get_bias([FC3_SIZE])
    fc3 = tf.nn.relu(tf.matmul(fc2 ,fc3_w) + fc3_b)
    if train: fc3 = tf.nn.dropout(fc3, 0.5)
    '''
    fc4_w = get_weight([FC2_SIZE,OUTPUT_NODE],regularizer)
    fc4_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc2,fc4_w) + fc4_b
    return y
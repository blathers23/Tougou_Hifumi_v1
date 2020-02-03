
import tensorflow as tf 
import pandas as pd

filenames=['board.csv']
testfilenames = ['board_test.csv']

def read_my_file_format(filename_queue):
    reader = tf.TextLineReader()
    _, record_string = reader.read(filename_queue)

    record_defaults = [[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]
  
    col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28=tf.decode_csv(record_string,record_defaults=record_defaults)

    board=tf.stack([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25])
    
    """
    ready_board=tf.to_float(not_ready_board)
    Ready_board=np.multiply(ready_board,1.0/12.0)
    R_board=np.reshape(Ready_board,(-1,1))
    board=tf.to_float(R_board)
    """
    label=tf.stack([col26,col27,col28])

    return board, label

def pd_reader(filename_queue):
 
    data_frame =  pd.read_csv(filename_queue, header=None)
    dataset = data_frame.values
    board = dataset[:, 0:25].astype(float)
    label = dataset[:, 25:] 

    return board, label


def input_pipeline(batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, 
                                                num_epochs=num_epochs,
                                                shuffle=True)

    board,label = read_my_file_format(filename_queue)

    board_batch, label_batch = tf.train.shuffle_batch([board, label],
                                                    batch_size = batch_size, 
                                                    capacity = 1200,
                                                    min_after_dequeue = 0,
                                                    num_threads = 300)


    return board_batch, label_batch

def test_input_pipeline(batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(testfilenames, 
                                                num_epochs=num_epochs,
                                                shuffle=True)

    board,label = read_my_file_format(filename_queue)

    board_batch, label_batch = tf.train.shuffle_batch([board, label],
                                                    batch_size = batch_size, 
                                                    capacity = 1200,
                                                    min_after_dequeue = 0,
                                                    num_threads = 300)


    return board_batch, label_batch


def main():
    input_pipeline(100, num_epochs=None)

if __name__ == '__main__':
    main()
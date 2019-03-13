import numpy as np
import tensorflow as tf


def max_detection(soft_mask, window_size, threshold):
    w_depth = window_size[1]
    w_height = window_size[2]
    w_width = window_size[3]

    sm_shape = np.shape(soft_mask)
    sm_depth = sm_shape[1]
    sm_height = sm_shape[2]
    sm_width = sm_shape[3]

    max_pool = tf.nn.max_pool3d(soft_mask, window_size, window_size, padding="SAME", data_format="NDHWC")

    conv_filter = np.ones([w_depth,w_height,w_width,1,1])

    upsampled = tf.nn.conv3d_transpose(
                            max_pool,
                            conv_filter.astype(np.float32),
                            [1,sm_depth,sm_height,sm_width,1],
                            window_size,
                            padding='SAME',
                            data_format='NDHWC',
                            name=None
                        )

    
    maxima = tf.equal(upsampled, soft_mask)
    maxima = tf.logical_and(maxima, soft_mask>=threshold)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        maxima = sess.run(maxima)


    return np.array(maxima[0,:,:,:,0], dtype=np.uint8)

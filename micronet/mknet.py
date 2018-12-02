from mala.networks import unet, conv_pass
import tensorflow as tf
import os
import json

def create_network(input_shape, name, run):

    tf.reset_default_graph()

    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)

    out, _, _ = unet(raw_batched, 10, 4, [[1,1,1],[1,2,2],[1,2,2]])

    lsds_batched, _ = conv_pass(
        out,
        kernel_sizes=[1],
        num_fmaps=10,
        activation=None)

    output_shape_batched = lsds_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip the batch dimension

    lsds = tf.reshape(lsds_batched, output_shape)
    soft_mask = lsds[9,:,:,:]
    derivatives = lsds[:9,:,:,:]

    gt_lsds = tf.placeholder(tf.float32, shape=output_shape)
    gt_soft_mask = gt_lsds[9,:,:,:]
    gt_derivatives = gt_lsds[:9,:,:,:]

    # Soft weights for binary mask
    binary_mask = tf.cast(gt_soft_mask > 0, tf.float32)
    loss_weights_soft_mask = tf.ones(binary_mask.get_shape())
    loss_weights_soft_mask += tf.multiply(binary_mask, tf.reduce_sum(binary_mask))
    loss_weights_soft_mask -= binary_mask
    #loss_weights_soft_mask /= tf.reduce_sum(loss_weights_soft_mask)

    loss_weights_lsds = tf.stack([loss_weights_soft_mask] * 10)

    loss = tf.losses.mean_squared_error(
                                lsds,
                                gt_lsds,
                                loss_weights_lsds)

    summary = tf.summary.scalar('loss', loss)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    output_shape = output_shape[1:]
    print("input shape : %s"%(input_shape,))
    print("output shape: %s"%(output_shape,))

    output_dir = "checkpoints/run_{}".format(run)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tf.train.export_meta_graph(filename=output_dir + "/" + name + '.meta')

    config = {
        'raw': raw.name,
        'lsds': lsds.name,
        'gt_lsds': gt_lsds.name,
        'loss_weights_lsds': loss_weights_soft_mask.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'summary': summary.name
    }

    with open(output_dir + "/" + name + '.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    create_network((32, 320, 320), 'micro_net', 8)

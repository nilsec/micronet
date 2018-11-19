from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
from lsd.gp import AddLocalShapeDescriptor
import os
import math
import json
import tensorflow as tf
import numpy as np

data_dir = './data'
samples = [
    'a+_master',
    'b+_master',
    'c+_master'
]

def train_until(max_iteration, snapshot_dir, checkpoint_dir, log_dir):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open(checkpoint_dir + '/micro_net.json', 'r') as f:
        config = json.load(f)

    raw = ArrayKey('RAW')
    tracing = ArrayKey('TRACING')
    lsds = ArrayKey('LSDS')
    gt_lsds = ArrayKey('GT_LSDS')
    loss_weights_soft_mask = ArrayKey('LOSS_WEIGHTS_SOFT_MASK')
    loss_weights_derivatives = ArrayKey('LOSS_WEIGHTS_DERIVATIVES')

    voxel_size = Coordinate((40, 4, 4))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(tracing, output_size)
    request.add(gt_lsds, output_size)
    
    snapshot_request = BatchRequest({
        lsds: request[tracing],
        loss_weights_soft_mask: request[tracing],
        loss_weights_derivatives: request[tracing]
        })

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.h5'),
            datasets = {
                raw: 'raw',
                tracing: 'tracing'
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
                tracing: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Pad(raw, None) +
        RandomLocation()
        for sample in samples
    )


    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8) +
        SimpleAugment(transpose_only=[1, 2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        AddLocalShapeDescriptor(
            tracing,
            gt_lsds,
            sigma=2.0,
            downsample=1) +
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            checkpoint_dir + '/micro_net',
            optimizer=config['optimizer'],
            loss=config['loss'],
            inputs={
                config['raw']: raw,
                config['gt_lsds']: gt_lsds
            },
            outputs={
                config['lsds']: lsds,
                config['loss_weights_soft_mask']: loss_weights_soft_mask,
                config['loss_weights_derivatives']: loss_weights_derivatives
            },
            gradients={},
            summary=config['summary'],
            log_dir=log_dir,
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'raw',
                tracing: 'tracing',
                gt_lsds: 'gt_lsds',
                lsds: 'lsds',
                loss_weights_soft_mask: 'loss_weights_soft_mask',
                loss_weights_derivatives: 'loss_weights_derivatives'
            },
            dataset_dtypes={
                tracing: np.uint64
            },
            every=1000,
            output_filename=snapshot_dir + '/batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    iteration = int(sys.argv[1])
    run = int(sys.argv[2])

    snapshot_dir = "snapshots/run_{}".format(run)
    checkpoint_dir = "checkpoints/run_{}".format(run)
    log_dir = "logs/run_{}".format(run)
    dirs = [snapshot_dir, checkpoint_dir, log_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    train_until(iteration, "run_{}".format(run), checkpoint_dir, log_dir)


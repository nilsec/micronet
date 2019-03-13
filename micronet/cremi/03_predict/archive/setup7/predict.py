from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys

def predict(
        setup,
        iteration,
        in_file,
        in_dataset,
        read_roi,
        out_file,
        out_dataset):

    run_dir = "/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/cremi/02_train/setup{}/checkpoints".format(setup)
    with open(os.path.join(run_dir, 
                           'micro_net.json'), 'r') as f:
        net_config = json.load(f)

    # voxels
    input_shape = Coordinate(net_config['input_shape'])
    output_shape = Coordinate(net_config['output_shape'])
    print(output_shape)
    context = (input_shape - output_shape)//2
    print("Context is %s"%(context,))

    # nm
    voxel_size = Coordinate((40, 4, 4))
    read_roi *= voxel_size
    context_nm = context*voxel_size
    input_size = input_shape*voxel_size
    output_size = output_shape*voxel_size
    
    write_roi = read_roi.grow(-context_nm, -context_nm)
    print("Read ROI in nm is %s"%read_roi)
    print("Write ROI in nm is %s"%write_roi)

    print("Read ROI in voxel space is {}".format(read_roi/voxel_size))
    print("Write ROI in voxel space is {}".format(write_roi/voxel_size))


    raw = ArrayKey('RAW')
    embedding = ArrayKey('EMBEDDING')

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(embedding, output_size)

    pipeline = (
        Hdf5Source(
            in_file,
            datasets = {
                raw: in_dataset
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
            }
        ) +
        Pad(raw, size=None) +
        Crop(raw, read_roi) +
        Normalize(raw) +
        IntensityScaleShift(raw, 2,-1) +
        Predict(
            os.path.join(run_dir, 'micro_net_checkpoint_%d'%iteration),
            inputs={
                net_config['raw']: raw
            },
            outputs={
                net_config['lsds']: embedding
            },
            graph=os.path.join(run_dir, 'micro_net.meta'),
            array_specs={embedding: ArraySpec(roi=write_roi, voxel_size=voxel_size)}
        ) +
        Hdf5Write(
            dataset_names={
                embedding: out_dataset,
            },
            output_filename=out_file
        ) +
        PrintProfilingStats(every=10) +
        Scan(chunk_request, num_workers=10)
    )

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")

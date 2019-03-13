from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys
from predict import predict
from candidate_extraction import max_detection
from evaluate_maxima import match
import h5py


def evaluate_all(setup=1,
                 run=0,
                 iteration=3*10**5,
                 window_size=[1,1,5,5,1],
                 threshold=0.5,
                 gt_maxima="/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/data/maxima_lsds_b+_validation_master.h5",
                 gt_maxima_dset="maxima",
                 voxel_size=np.array([40,4,4]),
                 distance_threshold=39,
                 skip_if_done=True):

    output_base_dir = "./{}/run_{}".format(iteration, run)
    lsd_output_file = output_base_dir + "/lsds_validation_B+.h5"

    # De Morgan:
    if not (os.path.isfile(lsd_output_file) and skip_if_done):
        print("Predict B+ validation with setup {} and run {}".format(setup, run))
        predict_validation_B_plus(setup,
                                  run,
                                  iteration,
                                  output_base_dir,
                                  lsd_output_file)
    else:
        print("Skip prediction, use " + lsd_output_file)

    
    pred_maxima = os.path.dirname(lsd_output_file) + "/maxima_" + lsd_output_file.split("/")[-1] 
    if not (os.path.isfile(pred_maxima) and skip_if_done):
        print("Extract maxima from " + lsd_output_file)
        extract_maxima(lsd_output_file,
                       window_size,
                       threshold,
                       pred_maxima)
    else:
        print("Skip maxima extraction, use " + pred_maxima)


    eval_result = os.path.dirname(pred_maxima) + "/evaluation_" + pred_maxima.split("/")[-1]    
    if not (os.path.isfile(eval_result) and skip_if_done):
        print("Evaluate " + pred_maxima)
        pred_maxima_dset = "maxima"
        evaluate(gt_maxima, gt_maxima_dset, 
                 pred_maxima, pred_maxima_dset, 
                 voxel_size=np.array([40,4,4]), 
                 distance_threshold=39)
    else:
        print("Nothing to be done. Evaluation in " + eval_result)


def predict_validation_B_plus(setup=1,
                              run=0,
                              iteration=3*10**5,
                              output_base_dir=None,
                              lsd_output_file=None,
                              output_dataset="prediction"):

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    roi_begin = (90,100,100) 
    roi_size = (30,1000,1000)

    roi = Roi(
        roi_begin,
        roi_size)

    data_dir = "/groups/funke/home/ecksteinn/data/mt_data/cremi/data/MTTraining_CremiTest"
    sample_base = "sample_{}_20160601.hdf5"
    sample = "B+"
    input_file = data_dir + "/" + sample_base.format(sample)
    input_dataset = "/volumes/raw"

    if output_base_dir or lsd_output_file is None:
        output_base_dir = "./{}/run_{}".format(iteration, run)
        lsd_output_file = output_base_dir + "/lsds_validation_B+.h5"

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    predict(
        setup,
        run,
        iteration,
        input_file,
        input_dataset,
        roi,
        lsd_output_file,
        output_dataset)


def extract_maxima(lsd_prediction,
                   window_size=[1,1,5,5,1],
                   threshold=0.5,
                   output_path="./maxima.h5"):

    with h5py.File(lsd_prediction, "r") as f:
        lsd_data = np.array(f["prediction"])

    soft_mask = lsd_data[9,:,:,:]
    maxima = max_detection(np.reshape(soft_mask, [1] + list(np.shape(soft_mask)) + [1]), window_size, threshold)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("maxima", data=np.array(maxima, dtype=np.uint8))


def evaluate(gt_maxima, gt_maxima_dset, pred_maxima, pred_maxima_dset, voxel_size=np.array([40,4,4]), distance_threshold=39):
    with h5py.File(gt_maxima, "r") as f:
        gt_maxima_data = np.array(f[gt_maxima_dset])

    with h5py.File(pred_maxima, "r") as f:
        pred_maxima_data = np.array(f[pred_maxima_dset])

    
    tp, fn, fp, canvas_fn, canvas_fp = match(gt_maxima_data, pred_maxima_data, voxel_size)

    with h5py.File(os.path.dirname(pred_maxima) + "/evaluation_" + pred_maxima.split("/")[-1], "w") as f:
        f.create_dataset("canvas_fn", data=np.array(canvas_fn, dtype=np.uint8))
        f.create_dataset("canvas_fp", data=np.array(canvas_fp, dtype=np.uint8))
        f.create_dataset("tp", data=tp)
        f.create_dataset("fn", data=fn)
        f.create_dataset("fp", data=fp)


if __name__ == "__main__":
    evaluate_all(skip_if_done=True)

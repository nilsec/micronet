from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys
from predict import predict


def predict_validation_B():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    read_begin = (50,0,0) 
    read_size = (160,3000,3000)

    read_roi = Roi(
        read_begin,
        read_size)

    data_dir = "/groups/funke/home/ecksteinn/data/mt_data/cremi/data/MTTest_CremiTraining_Unaligned"
    sample_base = "sample_{}_padded_20160501.hdf"
    sample = "B"
    
    setup = 7
    prediction = sample + "0"
    iteration = 3*10**5
    input_file = data_dir + "/" + sample_base.format(sample)
    input_dataset = "/volumes/raw"

    output_base_dir = "./{}".format(iteration)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    output_file = output_base_dir + "/lsds_{}.h5".format(prediction)
    output_dataset = "prediction"

    predict(
        setup,
        iteration,
        input_file,
        input_dataset,
        read_roi,
        output_file,
        output_dataset)


if __name__ == "__main__":
    predict_validation_B()

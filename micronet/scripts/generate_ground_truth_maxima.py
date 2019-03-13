from lsd.local_shape_descriptor import get_local_shape_descriptors
import h5py
import numpy as np
import os
from candidate_extraction import max_detection
from scipy.ndimage.morphology import binary_erosion

def get_lsds(sample,
             sigma,
             voxel_size):


    with h5py.File(sample, "r") as f:
        tracing = np.array(f["tracing"])

    lsd = get_local_shape_descriptors(tracing,
                                      sigma,
                                      voxel_size)


    with h5py.File(os.path.dirname(sample) + "/lsds_" + sample.split("/")[-1], "w") as f:
        f.create_dataset("lsds", data=lsd)


def get_maxima(gt_lsds):

    with h5py.File(gt_lsds, "r") as f:
        gt_lsds_data = np.array(f["lsds"])

    soft_mask = gt_lsds_data[9,:,:,:]
    maxima = max_detection(np.reshape(soft_mask, [1] + list(np.shape(soft_mask)) + [1]), [1,1,5,5,1], 0.5)

    with h5py.File(os.path.dirname(gt_lsds) + "/maxima_" + gt_lsds.split("/")[-1], "w") as f:
        f.create_dataset("maxima", data=np.array(maxima, dtype=np.uint8))


if __name__ == "__main__":
    """
    get_lsds("/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/data/b+_validation_master.h5",
            sigma=(4.0,) * 3,
            voxel_size=np.array([40,4,4]))
    """

    get_maxima("/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/data/lsds_b+_validation_master.h5")

import numpy as np
import h5py

def crop_dataset(input_file, 
                 input_dataset, 
                 x_lim, 
                 y_lim, 
                 z_lim, 
                 output_file,
                 output_dataset):

    f = h5py.File(input_file, "r")
    data = np.array(f[input_dataset])
    attrs = f[input_dataset].attrs.items()
    f.close()

    data = data[:,
                z_lim["min"]:z_lim["max"],
                y_lim["min"]:y_lim["max"],
                x_lim["min"]:x_lim["max"]]

    f = h5py.File(output_file, "w")
    dset = f.create_dataset(name=output_dataset, data=data)
    for attr in attrs:
        dset.attrs.create(attr[0], attr[1])
    f.close()


def lsd_prediction_to_softmask(lsd_prediction, output_file):
    f = h5py.File(lsd_prediction, "r")
    lsds = np.array(f["/prediction"])
    attrs = f["/prediction"].attrs.items()
    f.close()

    print attrs
    soft_mask = lsds[9,:,:,:]

    f = h5py.File(output_file, "w")
    dset = f.create_dataset(name="exported_data", data=soft_mask)
    for attr in attrs:
        dset.attrs.create(attr[0], attr[1])
    f.close()

if __name__ == "__main__":
    crop_dataset("/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/predictions/run_10/prediction_0.h5",
                 "/prediction",
                 x_lim={"min": 104, "max": 1354},
                 y_lim={"min": 104, "max": 1354},
                 z_lim={"min":1, "max": 126},
                 output_file="/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/predictions/run_10/softmask0_cropped.h5",
                 output_dataset="/exported_data")

    """
    lsd_prediction_to_softmask("/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/predictions/run_7/prediction3_cropped.h5", 
                               "/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/predictions/run_7/softmask3_cropped.h5")
    """

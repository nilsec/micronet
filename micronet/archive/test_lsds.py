from mt_lsd import get_descriptor
import h5py
import numpy as np

def test():
    tracing_volume = "./b+_master.h5"

    f = h5py.File(tracing_volume, "r")
    data = np.array(f["tracing"])
    f.close()

    ids = np.unique(data)

    lsd_stack = []
    for track_id in ids[:10]:
        print track_id
        mask = (data == track_id).astype(np.uint16)
        lsds = get_descriptor(mask, 2.0, np.array([5.,5.,50.]))
        lsd_stack.append(lsds)

    # Stack along first dimension:
    lsd_stack = np.stack(lsd_stack, axis=0)

    f = h5py.File("./lsd_stack", "w")
    f.create_dataset("lsds", data=lsd_stack)
    f.close()

if __name__ == "__main__":
    test()

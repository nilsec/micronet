import h5py
import neuroglancer
import numpy as np

voxel_size=np.array([4,4,40])

softmask = "/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/cremi/03_predict/setup7/300000/softmask_B1.h5"
softmask_dset = "/exported_data"
softmask_offset = np.array([46,46,65])

raw = "/groups/funke/home/ecksteinn/data/mt_data/cremi/data/MTTest_CremiTraining_Unaligned/sample_B_padded_20160501.hdf"
raw_dset = "/volumes/raw"
raw_offset = (0,0,0)

edge_cost = "/groups/funke/home/ecksteinn/Projects/microtubules/mtrack/mtrack/4b_candidates.h5"
edge_cost_dset = "/tracing"
edge_cost_offset = np.array([100,100,90])

neuroglancer.set_server_bind_address('0.0.0.0')

f_softmask = h5py.File(softmask, "r")
softmask_data = f_softmask[softmask_dset]

if raw is not None:
    f_raw = h5py.File(raw, "r")
    raw_data = f_raw[raw_dset]

if edge_cost is not None:
    f_edge_cost = h5py.File(edge_cost, "r")
    edge_cost_data = f_edge_cost[edge_cost_dset]

def add(s, a, name, offset, voxel_size, shader=None):
    if shader == 'rgb':
        shader="""void main() 
                  { emitRGB(vec3(toNormalized(getDataValue(0)), 
                                 toNormalized(getDataValue(1)), 
                                 toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a,
                offset=np.array(offset) * voxel_size,
                voxel_size=voxel_size
            ),
            **kwargs)

viewer = neuroglancer.Viewer()
with viewer.txn() as s: 
    add(s, softmask_data, "softmask", softmask_offset, voxel_size)

    if raw is not None:
        add(s, raw_data, "raw", raw_offset, voxel_size)

    if edge_cost is not None:
        add(s, edge_cost_data, "edge_cost", edge_cost_offset, voxel_size)
        

print(viewer)

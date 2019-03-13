import h5py
import neuroglancer
import numpy as np

#"/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/data/b+_master.h5"
layers =["/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/data/b+_validation_master.h5",
         "/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/data/lsds_b+_validation_master.h5",
         "/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/cremi/03_predict/setup01/300000/run_0/lsds_validation_B+.h5"]
layer_dsets = ["tracing", "lsds", "prediction"]
layer_offsets = [[0,0,0], [0,0,0], [0,0,0]]
channels = [None, [9], [9]]
voxel_size=np.array([4,4,40])

layer_data = []
for layer, dset in zip(layers, layer_dsets):
    layer_data.append(h5py.File(layer, "r")[dset])


neuroglancer.set_server_bind_address('0.0.0.0')

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

    i = 0
    for data, offset in zip(layer_data, layer_offsets):
        if channels[i] is not None:
            data = data[channels[i], :,:,:]

        print(np.shape(data))
        add(s, data, layers[i].split("/")[-1] + str(i), offset, voxel_size)
        i += 1

print(viewer)

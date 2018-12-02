import numpy as np
import operator
import h5py
from mtrack.preprocessing import nml_to_g1


def read_tracing(tracing):
    g1 = nml_to_g1(tracing, None)
    ccs = g1.get_components(2, None, return_graphs=True)

    tracks = []
    for cc in ccs:
        vertices = []
        edges = []
        vertex_to_position = {}

        for v in cc.get_vertex_iterator():
            vertices.append(int(v))
            vertex_to_position[int(v)] = np.array(cc.get_position(v))

        for e in cc.get_edge_iterator():
            edges.append((int(e.source()), int(e.target())))

        tracks.append([vertices, edges, vertex_to_position])

    return tracks

def add_raw_channel(source_h5,
                    source_dset,
                    target_h5,
                    target_dset,
                    volume_shape,
                    offset,
                    voxel_size=np.array([40,4,4])):

    f_source = h5py.File(source_h5, "r")
    source_data = f_source[source_dset]
    source_data = np.array(source_data)[offset[2]:volume_shape[0] + offset[2],
                                        offset[1]:volume_shape[1] + offset[1],
                                        offset[0]:volume_shape[2] + offset[0]]
    f_source.close()

    f_target = h5py.File(target_h5, "a")
    dset = f_target.create_dataset(target_dset, data=source_data)
    dset.attrs['resolution'] = voxel_size
    f_target.close()

def tracing_to_volume(tracing, 
                      volume_shape, 
                      offset,
                      voxel_size,
                      write_to=None):

    # Knossos has +1 offset:
    knossos_offset = offset + 1

    tracks = read_tracing(tracing)
    canvas = np.zeros(volume_shape, dtype=np.uint16)

    path_id = 1
    for track in tracks:
        path = interpolate(vertices=track[0], 
                           edges=track[1], 
                           vertex_to_position=track[2], 
                           voxel_size=voxel_size)

        canvas = draw(canvas, 
                      knossos_offset,
                      path,
                      path_id)

        path_id += 1

    if write_to is not None:
        f = h5py.File(write_to, "w")
        dset = f.create_dataset("tracing", data=canvas)
        dset.attrs['resolution'] = np.array(voxel_size, dtype=int)
        f.close()

    return canvas


def draw(canvas, offset, path, path_id):
    for point in path:
        point -= offset
        canvas[point[2], point[1], point[0]] = path_id

    return canvas

def interpolate(vertices, edges, vertex_to_position, voxel_size):
    """
    vertices: list of vertex indices in a path

    edges: tuples (id_x, id_y) with id_x, id_y in vertices.

    positions: dict from vertex ids to positions in voxel_space

    returns: A unique list of voxels interpolating the path.
    """

    interpolation = []
    for e in edges:
        p0 = vertex_to_position[e[0]]
        p1 = vertex_to_position[e[1]]
        line = dda3(start=p0, end=p1, scaling=voxel_size)
        interpolation.extend(line)

    return np.unique(interpolation, axis=0)

def dda_round(x):
    """
    Round to nearest integer.
    """
    return (x + 0.5).astype(int)

def dda3(start, end, scaling):
    """
    Linear interpolation between start and end
    using the dda algorithm in 3D. Interpolation
    performed on nm grid and downscaled to voxel grid
    afterwards.
    """
    # Scale to physical grid:
    start = np.array((start * scaling), dtype=float)
    end = np.array((end * scaling), dtype=float)
    assert(np.all(start.astype(int) == start))
    assert(np.all(end.astype(int) == end))

    max_direction, max_length = max(enumerate(abs(end - start)), key=operator.itemgetter(1))
    dv = (end - start)/max_length

    line = [dda_round(start/scaling)]
    for step in range(int(max_length)):
        point = np.array(dda_round(((step + 1) * dv + start)/scaling))
        if not np.all(point == line[-1]):
            line.append(point)

    assert(np.all(line[-1] == dda_round(end/scaling)))
    return line

def analyze_tracing(tracing):
    g1 = nml_to_g1(tracing, None)
    x = []
    y = []
    z = []

    for v in g1.get_vertex_iterator():
        pos = np.array(g1.get_position(v))
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])

    print "x: ", min(x), max(x)
    print "y: ", min(y), max(y)
    print "z: ", min(z), max(z)


def gen_a():
    # xy: 100 - 1100, z: 10 - 40
    tracing = "/groups/funke/home/ecksteinn/data/mt_data/cremi/tracings/a+_master.nml"
    volume_shape = np.array([30,1000,1000])
    offset = np.array([100, 100, 10])
    voxel_size = [40.,4.,4.]
    write_to = "./a+_master.h5"

    tracing_to_volume(tracing, 
                      volume_shape, 
                      offset,
                      voxel_size,
                      write_to=write_to)

    source_h5 = "/groups/funke/home/ecksteinn/data/mt_data/cremi/data/MTTraining_CremiTest/sample_A+_20160601.hdf5"
    source_dset = "/volumes/raw"

    target_h5 = write_to
    target_dset = "/raw"

    add_raw_channel(source_h5,
                    source_dset,
                    target_h5,
                    target_dset,
                    volume_shape,
                    offset)

def gen_b():
    # xy: 200 - 1200, z: 50 - 80
    tracing = "/groups/funke/home/ecksteinn/data/mt_data/cremi/tracings/b+_master.nml"
    volume_shape = np.array([30,1000,1000])
    offset = np.array([200, 200, 50])
    voxel_size = [40.,4.,4.]
    write_to = "./b+_master.h5"

    tracing_to_volume(tracing, 
                      volume_shape, 
                      offset,
                      voxel_size,
                      write_to=write_to)

    source_h5 = "/groups/funke/home/ecksteinn/data/mt_data/cremi/data/MTTraining_CremiTest/sample_B+_20160601.hdf5"
    source_dset = "/volumes/raw"

    target_h5 = write_to
    target_dset = "/raw"

    add_raw_channel(source_h5,
                    source_dset,
                    target_h5,
                    target_dset,
                    volume_shape,
                    offset)

    
def gen_c():
    # xy: 100 - 1100, z: 40 - 70
    tracing = "/groups/funke/home/ecksteinn/data/mt_data/cremi/tracings/c+_master.nml"
    volume_shape = np.array([30,1000,1000])
    offset = np.array([100, 100, 40])
    voxel_size = [40.,4.,4.]
    write_to = "./c+_master.h5"

    tracing_to_volume(tracing, 
                      volume_shape, 
                      offset,
                      voxel_size,
                      write_to=write_to)

    source_h5 = "/groups/funke/home/ecksteinn/data/mt_data/cremi/data/MTTraining_CremiTest/sample_C+_20160601.hdf5"
    source_dset = "/volumes/raw"

    target_h5 = write_to
    target_dset = "/raw"

    add_raw_channel(source_h5,
                    source_dset,
                    target_h5,
                    target_dset,
                    volume_shape,
                    offset)

def gen_validation_b():
    # xy: 100 - 1100, z: 90 - 120
    tracing = "/groups/funke/home/ecksteinn/data/mt_data/cremi/tracings/b+_validation_master.nml"
    volume_shape = np.array([30,1000,1000])
    offset = np.array([100,100,90])
    voxel_size = [40.,4.,4.]
    write_to = "./b+_validation_master.h5"

    tracing_to_volume(tracing,
                      volume_shape,
                      offset,
                      voxel_size,
                      write_to=write_to)

if __name__ == "__main__":
    gen_validation_b()
    #tracing = "/groups/funke/home/ecksteinn/data/mt_data/cremi/tracings/a+_master.nml"
    #analyze_tracing(tracing)
    #gen_a()
    #gen_b()
    #gen_c()

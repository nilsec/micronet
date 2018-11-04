import numpy as np
import operator
from scipy.spatial import KDTree


def build_kdtree(paths):
    """
    Assumes all points are unique, this is a reasonable 
    assumption for the groundtruth.

    Initializes a kdtree containing all positions of all paths
    and initializes a dictionary from positions to path id.

    This function should be called only once.

    returns: kdtree of points in paths, dict from points to path_id
    """
    
    points_to_path = {}
    all_points = []
    for path_id in range(len(paths)):
        path = paths[path_id]
        
        for pos in path:
            points_to_path[tuple(pos)] = path_id
            all_points.append(pos)

    paths_tree = KDTree(all_points)
    return paths_tree, points_to_path


def get_closest_path(voxel_coordinate, paths, paths_tree, points_to_path, voxel_size):
    """
    voxel_coordinate: numpy.array, the coordinate of the voxel to query in voxel space

    paths: list of lists [[p00, p01, p02, ...], [p10, p11, ...]] each list
           should contain the (interpolated) points in physical space corresponding to 
           a path.

    paths_tree: the KDtree of all paths to query against (in physical coordinates).

    returns: The list of points corresponding to the path with shortest euclidean distance to 
             the voxel_coordinate as calculated in physical space.
    """

    phyical_coordinate = voxel_coordinate * voxel_size
    nearest_neighbour = paths_tree.query(physical_coordinate)

    # Offset vector FROM coordinate TO nearest neigbour:
    # I.e. coord + offset = nn -> offset = nn - coord
    physical_offset = nearest_neighbour - physical coordinate

    path_id = points_to_path[tuple(nearest_neighbour)]
    return paths[path_id], nearest_neighbour, physical_offset


def get_covariance(path, point_on_path, radius):
    """
    path: list of points of the path in physical space

    point_on_path: The point around which the covariance should be calculated.
                   This point needs to be in the path and thus in physical space as well.

    radius: Radius in physical coordinates indicating the radius of the sphere in which 
            points on the path will be considered for calculation of the covariance.
    """

    points_in_roi = [p for p in point_on_path if (np.linalg.norm(p - point_on_path) <= radius)]
    points_in_roi = np.array(points_in_roi).T
    covariance = np.cov(points_in_roi)

    return covariance
    

def interpolate(vertices, edges, vertex_to_position, voxel_size):
    """
    vertices: list of vertex indices in a path

    edges: tuples (id_x, id_y) with id_x, id_y in vertices.

    positions: dict from vertex ids to positions in voxel_space

    returns: A list of points corresponding to the interpolated path 
             on a unit sized (1 unit defined by voxel_size) grid.
    """

    interpolation = []
    for e in edges:
        p0 = vertex_to_position[e[0]]
        p1 = vertex_to_position[e[1]]
        line = dda3(start=p0, end=p1, scaling=voxel_size)
        interpolation.extend(line)

    return interpolation

def dda_round(x):
    """
    Round to nearest integer.
    """
    return (x + 0.5).astype(int)

def dda3(start, end, scaling):
    """
    Linear interpolation between start and end
    using the dda algorithm in 3D. A step
    corresponds to one unit defined by scaling.
    """
    # Scale to physical grid:
    start = np.array((start * scaling), dtype=float)
    end = np.array((end * scaling), dtype=float)
    assert(np.all(start.astype(int) == start))
    assert(np.all(end.astype(int) == end))

    max_direction, max_length = max(enumerate(abs(end - start)), key=operator.itemgetter(1))
    dv = (end - start)/max_length

    line = [np.array(dda_round(start/scaling))]
    for step in range(int(max_lentgh)):
        point = np.array(dda_round(step + 1) * dv + start)
        line.append(point)

    assert(np.all(line[-1] == dda_round(end)))
    return line

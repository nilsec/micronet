import unittest
import numpy as np

from micronet.process import interpolate, build_kdtree, get_closest_path

class BaseTest(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.vertices = np.arange(self.n)
        self.start_end_vertices = [0, self.n - 1]
        self.edges = [(i,i+1) for i in range(self.n - 1)]
        self.vertex_to_position = {v: [v, 2*v, 0] for v in range(self.n)}
        self.voxel_size = np.array([5.,5.,50.])

class PathTest(unittest.TestCase):
    def setUp(self):
        self.voxel_size = np.array([5.,5.,50.])

        self.n = 10
        self.vertices = np.arange(self.n)
        self.start_end_vertices = [0, self.n - 1]
        self.edges = [(i,i+1) for i in range(self.n - 1)]
        self.vertex_to_position = {v: [v, 2*v, 0] for v in range(self.n)}

        self.path = interpolate(self.vertices,
                           self.edges,
                           self.vertex_to_position,
                           self.voxel_size)

        self.n1 = 12
        self.vertices1 = np.arange(self.n)
        self.start_end_vertices1 = [0, self.n - 1]
        self.edges1 = [(i,i+1) for i in range(self.n - 1)]
        self.vertex_to_position1 = {v: [v+1, v+1, 0] for v in range(self.n)}

        self.path1 = interpolate(self.vertices1,
                            self.edges1,
                            self.vertex_to_position1,
                            self.voxel_size)

        self.paths = [self.path, self.path1]


class InterpolateTestCase(BaseTest):
    def runTest(self):
        interpolation = interpolate(self.vertices,
                                    self.edges,
                                    self.vertex_to_position,
                                    self.voxel_size)

        self.assertTrue(len(set([tuple(p) for p in interpolation])) == len(interpolation))
        self.assertTrue(np.all(interpolation[0] == self.vertex_to_position[0] * self.voxel_size))
        self.assertTrue(np.all(interpolation[-1] == self.vertex_to_position[self.n - 1] * self.voxel_size))


class GetClosestPathTestCase(PathTest):
    def runTest(self):
        paths_tree, points_to_path = build_kdtree(self.paths)

        # Test points to path correctness:
        for pos in self.vertex_to_position.values():
            self.assertTrue(points_to_path[tuple(pos * self.voxel_size)] == 0)

        for pos in self.path:
            self.assertTrue(points_to_path[tuple(pos)] == 0)

        for pos in self.vertex_to_position1.values():
            self.assertTrue(points_to_path[tuple(pos * self.voxel_size)] == 1)

        for pos in self.path1:
            self.assertTrue(points_to_path[tuple(pos)] == 1)

        # Test get_closest_path correctness:
        for pos in self.vertex_to_position.values():
            path, nearest_neighbour, physical_offset = get_closest_path(pos, self.paths, paths_tree, points_to_path, self.voxel_size)
            print physical_offset


if __name__ == "__main__":
    unittest.main()

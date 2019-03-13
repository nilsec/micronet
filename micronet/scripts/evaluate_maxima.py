import numpy as np
import h5py
import networkx as nx
from scipy.spatial import KDTree


def match(gt_maxima, pred_maxima, voxel_size, distance_threshold=39):
    assert(np.all(np.shape(gt_maxima) == np.shape(pred_maxima)))

    gt_maxima_locations = np.array(np.nonzero(gt_maxima)).T
    pred_maxima_locations = np.array(np.nonzero(pred_maxima)).T

    vertices_gt = {i: gt_maxima_locations[i] for i in range(len(gt_maxima_locations))}
    vertices_pred = {j + len(gt_maxima_locations): pred_maxima_locations[j] for j in range(len(pred_maxima_locations))}
    vertices = vertices_gt.copy()
    vertices.update(vertices_pred)

    G = nx.Graph()
    for v, pos in vertices.iteritems():
        G.add_node(v, position=pos)


    gt_tree = KDTree(gt_maxima_locations * np.array(voxel_size))
    pred_tree = KDTree(pred_maxima_locations * np.array(voxel_size))

    results = gt_tree.query_ball_tree(pred_tree, r=distance_threshold)

    edges = []
    edge_distances = []
    for gt_id in range(len(results)):
        for pred_id in results[gt_id]:
            edges.append((gt_id, pred_id + len(gt_maxima_locations)))
            edge_distances.append(np.linalg.norm(gt_maxima_locations[gt_id] - pred_maxima_locations[pred_id]))


    eps = 10e-6
    for edge, distance in zip(edges, edge_distances):
        G.add_edge(edge[0], edge[1], weight=1./(distance + eps))

    pairs = nx.max_weight_matching(G, maxcardinality=True)


    v_gt = set([v for v in vertices_gt.keys()])
    v_pred = set([w for w in vertices_pred.keys()])
    true_positives = len(pairs)

    for edge in pairs:
        try:
            v_gt.remove(edge[0])
            v_pred.remove(edge[1])
        except KeyError:
            v_gt.remove(edge[1])
            v_pred.remove(edge[0])

    false_negatives = len(v_gt)
    false_positives = len(v_pred)

    
    canvas_fn = np.zeros(np.shape(gt_maxima))
    canvas_fp = np.zeros(np.shape(pred_maxima))
    for fn in v_gt:
        canvas_fn[vertices_gt[fn][0], vertices_gt[fn][1], vertices_gt[fn][2]] = 1
    for fp in v_pred:
        canvas_fp[vertices_pred[fp][0], vertices_pred[fp][1], vertices_pred[fp][2]] = 1

    return true_positives, false_negatives, false_positives, canvas_fn, canvas_fp

def test():
    f = h5py.File("/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/cremi/02_train/setup11/snapshots/run_0/batch_111501.hdf", "r")
    gt_maxima = np.array(f["gt_maxima"], dtype=np.float32)
    pred_maxima = np.array(f["pred_maxima"], dtype=np.float32)
    tp, fn, fp, canvas_fn, canvas_fp = match(gt_maxima, pred_maxima, np.array([40,4,4]))
    print tp, fn, fp

    f = h5py.File("eval_test.h5", "w")
    f.create_dataset("fn", data=canvas_fn)
    f.create_dataset("fp", data=canvas_fp)


def simple_test():
    gt_maxima = np.zeros([1,10,10])
    pred_maxima = np.zeros([1,10,10])

    gt_maxima[0,5,5] = 1
    gt_maxima[0,5,4] = 1
    pred_maxima[0,5,5] = 1
    pred_maxima[0,5,4] = 1
    tp, fn, fp = match(gt_maxima, pred_maxima, np.array([1,1,1]))
    print tp, fn, fp

if __name__ == "__main__":
    test()

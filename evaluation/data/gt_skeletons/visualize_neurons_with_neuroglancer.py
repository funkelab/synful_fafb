import itertools
import logging
import random
import sys

import neuroglancer
import numpy as np
import pandas as pd

neuroglancer.set_server_bind_address('0.0.0.0')
ngid = itertools.count(start=1)


def add_neuron(s, df, neuron_ids=[],
               show_ellipsoid_annotation=False):
    if len(neuron_ids) == 0:
        neuron_ids = list(np.unique(df.neuron_id))
    for ii, neuron_id in enumerate(neuron_ids):
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        pos_dic = {}
        df_neuron = df[df.neuron_id == neuron_id]
        edges = []
        for index, row in df_neuron.iterrows():
            node_id = row.id
            pos = np.array(row.position)
            pos_dic[node_id] = np.flip(pos)
            if row.parent_id:
                edges.append((node_id, row.parent_id))

        v_nodes, u_nodes, connectors = [], [], []
        print(f'Loaded {len(pos_dic)} nodes for neuron id {neuron_id}')
        for u, v in edges:
            if u in pos_dic and v in pos_dic:
                u_site = pos_dic[u]
                v_site = pos_dic[v]

                u_nodes.append(
                    neuroglancer.EllipsoidAnnotation(center=u_site,
                                                     radii=(
                                                         30, 30, 30),
                                                     id=next(ngid)))
                v_nodes.append(
                    neuroglancer.EllipsoidAnnotation(center=v_site,
                                                     radii=(
                                                         30, 30, 30),
                                                     id=next(ngid)))
                connectors.append(
                    neuroglancer.LineAnnotation(point_a=u_site,
                                                point_b=v_site,
                                                id=next(ngid)))

        s.layers['neuronskeleton_{}'.format(neuron_id)] = neuroglancer.AnnotationLayer(
            voxel_size=(1, 1, 1),
            filter_by_segmentation=False,
            annotation_color=color,
            annotations=connectors,
        )
        if show_ellipsoid_annotation:
            s.layers[
                'node_u_{}'.format(neuron_id)] = neuroglancer.AnnotationLayer(
                voxel_size=(1, 1, 1),
                filter_by_segmentation=False,
                annotation_color=color,
                annotations=u_nodes,
            )
            s.layers[
                'node_v_{}'.format(neuron_id)] = neuroglancer.AnnotationLayer(
                voxel_size=(1, 1, 1),
                filter_by_segmentation=False,
                annotation_color=color,
                annotations=v_nodes,
            )


if __name__ == '__main__':
    """
    Script to display FAFB neurons.
    Example Usage: python -i visualize_neurons_with_neuroglancer.py lh_skeletons.json 23829,38885
    
    If no neuron_ids are provided, all neurons in the given file are displayed:
    python -i visualize_neurons_with_neuroglancer.py lh_skeletons.json
    
    """

    logging.basicConfig(level=logging.INFO)

    filename = sys.argv[1]
    if len(sys.argv) > 2:
        neuron_ids = [int(neuron_id) for neuron_id in sys.argv[2].split(',')]
    else:
        neuron_ids = []
    df_skels = pd.read_json(filename)

    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.layers['raw'] = neuroglancer.ImageLayer(
            source='precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig',
        )
        s.layers['ffn1'] = neuroglancer.SegmentationLayer(
            source='precomputed://gs://fafb-ffn1-20190805/segmentation',
        )

        add_neuron(s, df_skels, neuron_ids)

    print(viewer.__str__())
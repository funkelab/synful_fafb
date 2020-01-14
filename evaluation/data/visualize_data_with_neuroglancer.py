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

        s.layers['neuronskeleton_{}'.format(
            neuron_id)] = neuroglancer.AnnotationLayer(
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


def add_synapses(s, df, pre_neuron_id=None, post_neuron_id=None, color=None,
                 name='', radius=30):
    if pre_neuron_id is not None:
        df = df[df.id_skel_pre == pre_neuron_id]
    if post_neuron_id is not None:
        df = df[df.id_skel_post == post_neuron_id]

    pre_sites, post_sites, connectors = [], [], []
    print('Displaying {} of synapses'.format(len(df)))
    for index, syn in df.iterrows():
        pre_site = np.flip(syn.location_pre)
        post_site = np.flip(syn.location_post)

        pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
                                                          radii=(
                                                              radius,
                                                              radius,
                                                              radius),
                                                          id=next(ngid)))
        post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
                                                           radii=(
                                                               radius,
                                                               radius,
                                                               radius),
                                                           id=next(ngid)))
        connectors.append(
            neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                        id=next(ngid)))

    s.layers['synlinks_{}'.format(name)] = neuroglancer.AnnotationLayer(
        voxel_size=(1, 1, 1),
        filter_by_segmentation=False,
        annotation_color=color,
        annotations=connectors,
    )


if __name__ == '__main__':
    """
    Script to display FAFB neurons.
    Example Usage: python -i visualize_neurons_with_neuroglancer.py gt_skeletons/lh_skeletons.json lhfull_gt.json 23829
    """

    logging.basicConfig(level=logging.INFO)

    filename = sys.argv[1]
    synapse_filename = sys.argv[2]
    neuron_id = int(sys.argv[3])

    df_skels = pd.read_json(filename)
    df_synapses = pd.read_json(synapse_filename)

    input_site_color = '#72b9cb'
    output_site_color = '#c12430'

    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.layers['raw'] = neuroglancer.ImageLayer(
            source='precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig',
        )
        s.layers['ffn1'] = neuroglancer.SegmentationLayer(
            source='precomputed://gs://fafb-ffn1-20190805/segmentation',
        )

        add_neuron(s, df_skels, [neuron_id])
        add_synapses(s, df_synapses, pre_neuron_id=neuron_id,
                     color=output_site_color, name='output')
        add_synapses(s, df_synapses, post_neuron_id=neuron_id,
                     color=input_site_color, name='input')

    print(viewer.__str__())

import csv
import json
import logging
import multiprocessing as mp
import os
import sys

import daisy
import numpy as np
import pandas as pd
import pymaid
import scipy

from synful import database, synapse, evaluation

logger = logging.getLogger(__name__)


def csv_to_list(csvfilename, column):
    with open(csvfilename) as csvfile:
        data = list(csv.reader(csvfile))
    col_list = []
    for ii in range(column, len(data)):
        row = data[ii]
        col_list.append(int(row[column]))
    return col_list

class EvaluateFafb():
    '''Evaluation class for evaluating predicted synapses against ground truth
    synapses mapped onto skeletons. Evaluation is based on Hungarian matching,
    each matched predicted synapse is considered True Positive, unmatched
    predicted synapse False Positive and unmatched ground-truth synapse
    False Negative. Only synapses are matched in the first place, if they have
    the right underlying connectivity.

    Args:
        pred_synlinks (`str`): Json file path to predicted synapses.
        gt_synlinks (`str`): Json file path to ground truth synapses.
        results_dir (`str`): Directory to write results jsonfiles to.
    '''

    def __init__(self, pred_synlinks,
                 gt_synlinks, results_dir=None,
                 multiprocess=True, matching_threshold=400,
                 matching_threshold_only_post=False,
                 matching_threshold_only_pre=False,
                 skeleton_ids=None,
                 filter_same_id=False, filter_same_id_type='seg',
                 filter_redundant=False,
                 filter_redundant_dist_thr=None, filter_redundant_id_type='seg',
                 only_input_synapses=False,
                 only_output_synapses=False,
                 roi_file=None, syn_dir=None,
                 filter_redundant_dist_type='euclidean',
                 filter_redundant_ignore_ids=[], filter_seg_ids=[]):
        assert filter_redundant_id_type == 'seg' or filter_redundant_id_type == 'skel'
        assert filter_same_id_type == 'seg' or filter_same_id_type == 'skel'
        assert filter_redundant_dist_type == 'euclidean' or \
               filter_redundant_dist_type == 'geodesic'
        self.pred_synlinks = pred_synlinks
        self.gt_synlinks = gt_synlinks

        self.matching_threshold = matching_threshold
        self.skeleton_ids = skeleton_ids

        self.multiprocess = multiprocess

        assert not (
                only_input_synapses is True and only_output_synapses is True), 'both only_input_synapses and only_output_synapses is set to True, unclear what to do'
        # Evaluation settings
        self.filter_same_id = filter_same_id
        self.filter_redundant = filter_redundant
        self.filter_redundant_dist_thr = filter_redundant_dist_thr
        self.only_input_synapses = only_input_synapses
        self.only_output_synapses = only_output_synapses
        self.matching_threshold_only_post = matching_threshold_only_post
        self.matching_threshold_only_pre = matching_threshold_only_pre
        # Where to write out results to
        self.results_dir = results_dir
        self.roi_file = roi_file
        self.syn_dir = syn_dir
        self.filter_same_id_type = filter_same_id_type
        self.filter_redundant_id_type = filter_redundant_id_type
        self.filter_redundant_dist_type = filter_redundant_dist_type
        self.filter_redundant_ignore_ids = filter_redundant_ignore_ids
        self.filter_seg_ids = filter_seg_ids

        # Load synapses
        self.pred_df = pd.read_json(pred_synlinks)
        self.pred_df = self.pred_df.replace({pd.np.nan: None})
        self.gt_df = pd.read_json(gt_synlinks)
        self.gt_df = self.gt_df.replace({pd.np.nan: None})

    def get_cremi_score(self, score_thr=0, skel_ids=None):

        if skel_ids is None:
            assert self.skeleton_ids is not None
            skel_ids = csv_to_list(self.skeleton_ids, 0)
        else:
            assert self.skeleton_ids is None
            assert type(skel_ids) is list

        fpcountall, fncountall, predall, gtall, tpcountall, num_clustered_synapsesall = 0, 0, 0, 0, 0, 0

        pred_synapses_all = []
        for skel_id in skel_ids:
            logger.debug('evaluating skeleton {}'.format(skel_id))
            if not self.only_output_synapses and not self.only_input_synapses:
                pred_synapses = self.pred_df[(self.pred_df.id_skel_pre == skel_id) | (self.pred_df.id_skel_post == skel_id)]
                gt_synapses = self.gt_df[(self.gt_df.id_skel_pre == skel_id) | (self.gt_df.id_skel_post == skel_id)]

            elif self.only_input_synapses:
                pred_synapses = self.pred_df[self.pred_df.id_skel_post == skel_id]
                gt_synapses = self.gt_df[self.gt_df.id_skel_post == skel_id]
            elif self.only_output_synapses:
                pred_synapses = self.pred_df[self.pred_df.id_skel_pre == skel_id]
                gt_synapses = self.gt_df[self.gt_df.id_skel_pre == skel_id]
            else:
                raise Exception(
                    'Unclear parameter configuration: {}, {}'.format(
                        self.only_output_synapses, self.only_input_synapses))

            pred_synapses = [synapse.Synapse(**dic) for dic in pred_synapses.to_dict(orient='records')]

            if not len(self.filter_seg_ids) == 0:
                pred_synapses = [syn for syn in pred_synapses if not (
                        syn.id_segm_pre in self.filter_seg_ids or syn.id_segm_post in self.filter_seg_ids)]

            pred_synapses = [syn for syn in pred_synapses if
                             syn.score >= score_thr]
            if self.filter_same_id:
                if self.filter_same_id_type == 'seg':
                    pred_synapses = [syn for syn in pred_synapses if
                                     syn.id_segm_pre != syn.id_segm_post]
                elif self.filter_same_id_type == 'skel':
                    pred_synapses = [syn for syn in pred_synapses if
                                     syn.id_skel_pre != syn.id_skel_post]
            if self.filter_redundant:
                assert self.filter_redundant_dist_thr is not None
                num_synapses = len(pred_synapses)
                if self.filter_redundant_dist_type == 'geodesic':
                    # Get skeleton
                    skeleton = pymaid.get_neurons([skel_id])
                else:
                    skeleton = None
                __, removed_ids = synapse.cluster_synapses(pred_synapses,
                                                           self.filter_redundant_dist_thr,
                                                           fuse_strategy='max_score',
                                                           id_type=self.filter_redundant_id_type,
                                                           skeleton=skeleton,
                                                           ignore_ids=self.filter_redundant_ignore_ids)
                pred_synapses = [syn for syn in pred_synapses if
                                 not syn.id in removed_ids]
                num_clustered_synapses = num_synapses - len(pred_synapses)
                logger.debug(
                    'num of clustered synapses: {}, skel id: {}'.format(
                        num_clustered_synapses, skel_id))
            else:
                num_clustered_synapses = 0

            logger.debug(
                'found {} predicted synapses'.format(len(pred_synapses)))

            gt_synapses = [synapse.Synapse(**dic) for dic in gt_synapses.to_dict(orient='records')]
            stats = evaluation.synaptic_partners_fscore(pred_synapses,
                                                        gt_synapses,
                                                        matching_threshold=self.matching_threshold,
                                                        all_stats=True,
                                                        use_only_pre=self.matching_threshold_only_pre,
                                                        use_only_post=self.matching_threshold_only_post)
            fscore, precision, recall, fpcount, fncount, tp_fp_fn_syns = stats


            # tp_syns, fp_syns, fn_syns_gt, tp_syns_gt = evaluation.from_synapsematches_to_syns(
            #     matches, pred_synapses, gt_synapses)
            tp_syns, fp_syns, fn_syns_gt, tp_syns_gt = tp_fp_fn_syns
            fpcountall += fpcount
            fncountall += fncount
            tpcountall += len(tp_syns_gt)
            predall += len(pred_synapses)
            gtall += len(gt_synapses)
            num_clustered_synapsesall += num_clustered_synapses

            assert len(fp_syns) == fpcount
            pred_synapses_all.extend(pred_synapses)
            logger.info(
                f'skel id {skel_id} with fscore {float(fscore):0.2}, precision: {float(precision):0.2}, recall: {float(recall):0.2}')
            logger.info(f'fp: {fpcount}, fn: {fncount}')
            logger.info(f'total predicted {len(pred_synapses)}; total gt: {len(gt_synapses)}\n')

        pred_dic = {}
        for syn in pred_synapses_all:
            pred_dic[syn.id] = syn
        logger.debug('Number of duplicated syn ids: {} versus {}'.format(
            len(pred_synapses_all), len(pred_dic)))

        precision = float(tpcountall) / (tpcountall + fpcountall) if (
                                                                             tpcountall + fpcountall) > 0 else 0.
        recall = float(tpcountall) / (tpcountall + fncountall) if (
                                                                          tpcountall + fncountall) > 0 else 0.
        if (precision + recall) > 0:
            fscore = 2.0 * precision * recall / (precision + recall)
        else:
            fscore = 0.0

        # Collect all in a single document in order to enable quick queries.
        result_dic = {}
        result_dic['fscore'] = fscore
        result_dic['precision'] = precision
        result_dic['recall'] = recall
        result_dic['fpcount'] = fpcountall
        result_dic['fncount'] = fncountall
        result_dic['tpcount'] = tpcountall
        result_dic['predcount'] = predall
        result_dic['gtcount'] = gtall
        result_dic['score_thr'] = score_thr

        settings = {}
        settings['pred_synlinks'] = self.pred_synlinks
        settings['gt_synlinks'] = self.gt_synlinks

        settings['filter_same_id'] = self.filter_same_id
        settings['filter_same_id_type'] = self.filter_same_id_type
        settings['filter_redundant'] = self.filter_redundant
        settings['filter_redundant_id_type'] = self.filter_redundant_id_type
        settings['dist_thr'] = self.filter_redundant_dist_thr
        settings['skel_ids'] = self.skeleton_ids
        settings['matching_threshold'] = self.matching_threshold
        settings[
            'matching_threshold_only_post'] = self.matching_threshold_only_post
        settings[
            'matching_threshold_only_pre'] = self.matching_threshold_only_pre
        settings['only_output_synapses'] = self.only_output_synapses
        settings['only_input_synapses'] = self.only_input_synapses
        settings['num_clustered_synapses'] = num_clustered_synapsesall
        settings['filter_redundant_dist_type'] = self.filter_redundant_dist_type
        settings['filter_seg_ids'] = str(self.filter_seg_ids)

        result_dic.update(settings)
        if self.results_dir is not None:
            resultsfile = self.results_dir + 'results_thr{}.json'.format(1000 * score_thr)
            logger.info('writing results to {}'.format(resultsfile))
            with open(resultsfile, 'w') as f:
                json.dump(result_dic, f)

        print('final fscore {:0.2}'.format(fscore))
        print('final precision {:0.2}, recall {:0.2}'.format(precision, recall))
        return result_dic

    def evaluate_synapse_complete(self, score_thresholds):
        if self.multiprocess:
            pool = mp.Pool(10)
            pool.map(self.get_cremi_score, score_thresholds)
            pool.close()
            pool.join()
        else:
            for score_thr in score_thresholds:
                self.get_cremi_score(score_thr)

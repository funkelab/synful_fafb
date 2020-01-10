# Automatic Detection of Synaptic Partners in a Whole-Brain Drosophila EM Dataset

This repository serves as an entry point for accessing and interacting with
predicted synaptic partners in the full adult fly brain (FAFB) dataset.

![method_figure](docs/_static/fafb_zoom_sequence.jpg)

Details about the method can be found in our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2019.12.12.874172v1).

If you are interested in

- using predicted synapses for circuit reconstruction in CATMAID, see the
  [CircuitMap](http://github.com/unidesigner/circuitmap) application

- downloading all 244 million predicted synaptic connections in FAFB, see [this
  SQL dump](https://cremi.org/static/data/20191211_fafbv14_buhmann2019_li20190805.db)
  and this [example jupyter notebook](https://github.com/flyconnectome/fafbseg-py/blob/query-synapses/notebooks/Synaptic_Partner_Predictions_in_FAFB.ipynb) on how to use it

- using our evaluation data to compare your own synapse prediction method, stay tuned for our data release (end of December 2019)!

- training and/or predicting on your own data, stay tuned for our source code release (end of December 2019)!

Please don't hesitate to open
an issue or write us an email ([Julia
Buhmann](mailto:buhmannj@janelia.hhmi.org) or [Jan
Funke](mailto:funkej@janelia.hhmi.org)) if you have any questions!

## Benchmark dataset and evaluation
---- work in progress -----

### Ground-truth datasets

- Neuron skeletons are available in this repos: `evaluation/data/gt_skeletons/<brain_region>_skeletons.json`
- Ground-truth synaptic links are available in this repos:
`evaluation/data/<brain_region>full_gt.json`

| Dataset | Connection count |     Brain Region     | Source                                             |
|---------|------------------|:--------------------:|----------------------------------------------------|
| OutLH   |      11,429      |     lateral horn     | Bates et al. (2019), in preparation (Jefferis Lab) |
| InOutEB |      61,280      |    ellipsoid body    | Turner-Evans et al. (2019) (Jayaraman Lab)         |
| InOutPB |      14,779      | protocerebral bridge | Turner-Evans et al. (2019) (Jayaraman Lab)         |

For more details on the datasets, please refer to our preprint, Table 1 and section `3.4.1 Evaluation:Datasets`.



### Evaluation

Evaluation code depends on the synful package. Please install from [synful repos](https://github.com/funkelab/synful).

We also added our predicted synful-synapses as example files.
Run evaluation on synful-synapses:
```shell
cd evaluation
python run_evaluate.py configs/pb_eval.json
```
This should output:
```
final fscore 0.59
final precision 0.62, recall 0.57
```

To test your own predicted synapses:

1) Predict synapses in the three brain regions in FAFB for which ground-truth is available
2) Map predicted synapses onto ground-truth skeletons provided in this repos
3) Write synapses out into the here required format, see this [section](Synapse-Format)
4) Adapt the config file and replace `pred_synlinks` with your predicted synapse-filepath (this [line](https://github.com/funkelab/synful_fafb/blob/master/evaluation/configs/eb_eval.json#L2) in the config file).

##### Synapse Format
Synapses are stored in a json file, each synapse is represented with:
```python
{"id": 822988374080568,
"id_skel_pre": 4429537,
"id_skel_post": 4210786,
"location_pre": [89200, 159400, 512924],
"location_post": [89200, 159412, 512816],
"score": 8.599292755126953
```
See [this file](https://raw.githubusercontent.com/funkelab/synful_fafb/master/evaluation/data/ebfull_gt.json) for an example of predicted synapses stored in the required format.
Locations are provided in physical units (nm) and z,y,x order.

##### Neuron Format
Neurons are represented with a list of nodes. One example node:
```python
{"id": 1760159,
"position": [171240, 157706, 482924],
"neuron_id": 1274114,
"parent_id": 19713274}
```
See [this file](evaluation/data/gt_skeletons/eb_skeletons.json) for an example of ground truth neurons.
Locations are provided in physical units (nm) and z,y,x order.
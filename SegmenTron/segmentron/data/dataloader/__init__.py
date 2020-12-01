"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .datasetA import DatasetA
from .datasetA_aug import AugDatasetA
from .datasetB_aug import AugDatasetB
from .datasetC import NaicrsDatasetC
from .datasetC_aug import AugDatasetC
from .datasetAC import DatasetAC
from .datasetAC_distill import DistillDatasetAC
from .datasetAC_filter import FilterDatasetAC

datasets = {
    'datasetA': DatasetA,
    'datasetA_aug': AugDatasetA,
    'datasetB_aug': AugDatasetB,
    'datasetC': NaicrsDatasetC,
    'datasetC_aug': AugDatasetC,
    'datasetAC': DatasetAC,
    'datasetAC_distill': DistillDatasetAC,
    'datasetAC_filter': FilterDatasetAC,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name](**kwargs)

from .simple_dataloaders import precache_dataloader
from .cifar_dataloaders import cifar_ffcv, cifar_classifier_ffcv, cifar_pt
from .stl10_dataloaders import stl_ffcv, stl10_pt, stl_classifier_ffcv 
from .cifar_transforms import CifarTransform, CifarClassifierTransform, CifarTransformFFCV, STLTransformFFCV
from .custom_ffcv_transforms import ColorJitter, RandomGrayscale


# basic mapping from dataset name and stage to dataloader
DATALOADER_REGISTRY = {
    "cifar10_pretrain_ffcv": cifar_ffcv,
    "cifar10_classifier_ffcv": cifar_classifier_ffcv,
    "cifar100_pretrain_ffcv": cifar_ffcv,
    "cifar100_classifier_ffcv": cifar_classifier_ffcv,
    "stl10_pretrain_ffcv": stl_ffcv,
    "stl10_classifier_ffcv": stl_classifier_ffcv,
}
from .raw_dataset import RawDataset
from .dataset import Dataset
from .ml100k import ML100k, ImplicitML100k, ExplicitML100k
from .ml1m import ML1M, ExplicitML1M, ImplicitML1M
from .ml10m import ML10M, ExplicitML10M, ImplicitML10M
from .ml20m import ML20M, ExplicitML20M, ImplicitML20M
from .ml25m import ML25M, ExplicitML25M, ImplicitML25M
from .ml100k_leakage import (
    ML100kLeakage, ExplicitML100kLeakage, ImplicitML100kLeakage
)
from .ml100k_negative_leakage import (
    ML100kNegativeLeakage,
    ExplicitML100kNegativeLeakage,
    ImplicitML100kNegativeLeakage
)
from .netflix import Netflix
from .netflix_transfer import (
    NetflixTransfer,
    ExplicitNetflixTransfer,
    ImplicitNetflixTransfer
)

name2cstrc = {
    'ML100k': ML100k,
    'ml100k': ExplicitML100k,
    'ML1M': ML1M,
    'ml1m': ExplicitML1M,
    'ML10M': ML10M,
    'ML20M': ML20M,
    'ML25M': ML25M,
    'Netflix': Netflix,
    'ML100kLeakage': ML100kLeakage,
    'ML100kNegativeLeakage': ML100kNegativeLeakage,
    'ExplicitML100kLeakage': ExplicitML100kLeakage,
    'ExplicitML100kNegativeLeakage': ExplicitML100kNegativeLeakage,
    'ImplicitML100kLeakage': ImplicitML100kLeakage,
    'ImplicitML100kNegativeLeakage': ImplicitML100kNegativeLeakage
}

DATASETS = [
    ML100k,
    ML1M,
    ML10M,
    ML20M,
    ML25M,
    Netflix
]

EXPLICIT_DATASETS = [
    ExplicitML100k,
    ExplicitML1M,
    ExplicitML10M,
    ExplicitML20M,
    ExplicitML25M,
    ExplicitML100kLeakage,
    ExplicitML100kNegativeLeakage
]

IMPLICIT_DATASETS = [
    ImplicitML100k,
    ImplicitML1M,
    ImplicitML10M,
    ImplicitML20M,
    ImplicitML25M,
    ImplicitML100kLeakage,
    ImplicitML100kNegativeLeakage
]


def load_from_name(name, **kwargs) -> Dataset:
    try:
        return name2cstrc[name]
    except KeyError:
        raise Exception('Dataset not yet implemented.')

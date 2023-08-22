from . import data_module
from . import image_dataset
from . import pretraining_dataset

DATA_MODULES = {
    "pretrain": data_module.PretrainingDataModule,#private
    "INB": data_module.INBDataModule,#inbreast, cbis, ddsm
}

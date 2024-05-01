from geqtrain.data import AtomicDataDict

RMSE_LOSS_KEY = "rmse"
MAE_KEY = "mae"
LOSS_KEY = "noramlized_loss"

VALUE_KEY = "value"
CONTRIB = "contrib"

VALIDATION = "validation"
TRAIN = "training"

ABBREV = {
    AtomicDataDict.NODE_FEATURES_KEY: "h",
    LOSS_KEY: "loss",
    VALIDATION: "val",
    TRAIN: "train",
}

"""
configs/model_config.py

Torchvision Mask R-CNN configuration for dental panoramic X-ray segmentation.

Dataset: OdontoAI - 1597 train images, 400 validation and 2000 test JPEG panoramic X-ray images
Classes: 52 FDI classes
         Adult permanent: 11-48 (32 classes)
         Deciduous/primary: 51-85 (20 classes)
"""

NUM_CLASSES  = 53            # set to 53, 32 FDI classes for adult tooth + 20 deciduous + background

# torchvision handles resize internally — 800 min  size is default
IMAGE_MIN_SIZE = 800        # shorter side
IMAGE_MAX_SIZE = 1333       # longer side — torchvision default

EPOCHS          = 35
BATCH_SIZE      = 2         # increase if VRAM allows
NUM_WORKERS     = 4
LR              = 0.001     # SGD learning rate
MOMENTUM        = 0.9
WEIGHT_DECAY    = 0.0005
LR_STEP_SIZE    = 10       # decay LR every N epochs
LR_GAMMA        = 0.1       # multiply LR by this at each step

# Higher threshold, higher precision, but lower recall (might miss true ones)
CONF_THRESHOLD  = 0.3 #0.5       # minimum score to show detection

# tooth are naturally closer, so better higher number (as adjescent teeth can overlap)
NMS_THRESHOLD   = 0.4 #0.3       # IOU threshold for NMS
MAX_DETECTIONS  = 60        # max instances per image

# After resize to ~800px shorter side, teeth are ~50-150px
ANCHOR_SIZES    = ((32,), (64,), (128,), (256,), (512,))
ANCHOR_RATIOS   = ((0.5, 1.0, 2.0),) * 5

EARLY_STOPPING_PATIENCE = 10

FDI_CLASSES = [
    '__background__',
    'tooth-11', 'tooth-12', 'tooth-13', 'tooth-14', 'tooth-15', 'tooth-16', 'tooth-17', 'tooth-18',
    'tooth-21', 'tooth-22', 'tooth-23', 'tooth-24', 'tooth-25', 'tooth-26', 'tooth-27', 'tooth-28',
    'tooth-31', 'tooth-32', 'tooth-33', 'tooth-34', 'tooth-35', 'tooth-36', 'tooth-37', 'tooth-38',
    'tooth-41', 'tooth-42', 'tooth-43', 'tooth-44', 'tooth-45', 'tooth-46', 'tooth-47', 'tooth-48',
    'tooth-51', 'tooth-52', 'tooth-53', 'tooth-54', 'tooth-55',
    'tooth-61', 'tooth-62', 'tooth-63', 'tooth-64', 'tooth-65',
    'tooth-71', 'tooth-72', 'tooth-73', 'tooth-74', 'tooth-75',
    'tooth-81', 'tooth-82', 'tooth-83', 'tooth-84', 'tooth-85',
]
# Class-Imbalanced Semi-Supervised Learning
* Code For Object Detection
* We referred to the following paper and code.
* [CSD: Consistency-based Semi-supervised learning for object Detection](https://github.com/soo89/CSD-SSD)

## Installation & Preparation
We experimented Suppressed Consistency Loss (SCL) on CSD using the SSD pytorch framework. To use our model, complete the installation & preparation on the [SSD pytorch homepage](https://github.com/amdegroot/ssd.pytorch)

#### prerequisites
- Python 3.6+
- Pytorch 1.1.0

## Supervised learning
```
python train_ssd.py
```

## CSD training
```
python train_csd.py
```

## CSD with SCL training
```
python train_csd_scl.py
```

## Evaluation
```
python eval.py
```

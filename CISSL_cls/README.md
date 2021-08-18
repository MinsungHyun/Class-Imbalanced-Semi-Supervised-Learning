# Class-Imbalanced Semi-Supervised Learning
* Code For Classification

## Requirements
- CUDA-enabled GPU
- Python 3.6+
- PyTorch 1.1.0
- torchvision 0.3.0
- numpy 1.16.2


## Prepare dataset (CIFAR10, SVHN)

```
sh build_dataset.sh
```

## Toy examples
* Twomoons, Fourspins (Fig.1)

```
sh run_toy.sh
```

## Experiments
#### Comparison of Imbalance Factor and Number of Labeled Samples
* CIFAR10 nlabels 4000, imbalance factor 100, seed 0 (Table.2a, Table.4a) 

```
sh run_cifar10.sh
```


* SVHN nlabels 1000, imbalance factor 100, seed 0 (Table.2b, Table.4b)

```
sh run_svhn.sh
```


#### Comparison of Class Imbalanced Learning Methods
* CIFAR10 nlabels 4000, imbalance factor 100, seed 0 (Table.3a) 
```
sh run_cifar10_reweight.sh
```


* SVHN nlabels 1000, imbalance factor 100, seed 0 (Table.3b)

```
sh run_svhn_reweight.sh
```

You can run expemerimens with different settings by changing arguments.

The size of unlabled data for each run is described in the supplementary material.

***

Please check the detailed options by 

```
python train_imbalance.py -h
```

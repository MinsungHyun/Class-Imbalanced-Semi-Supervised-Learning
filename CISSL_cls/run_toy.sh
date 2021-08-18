#!/usr/bin/env bash

# twomoons imbalanced
python toy_examples.py --seed 6 --dataset 'twomoons' --nlabels 12 --imb-factor 5 --alg 'supervised'
python toy_examples.py --seed 6 --dataset 'twomoons' --nlabels 12 --imb-factor 5 --alg 'MT'
python toy_examples.py --seed 6 --dataset 'twomoons' --nlabels 12 --imb-factor 5 --alg 'PI'
python toy_examples.py --seed 6 --dataset 'twomoons' --nlabels 12 --imb-factor 5 --alg 'MT' --scl 0.5

# fourspins imbalanced
python toy_examples.py --seed 0 --dataset 'fourspins' --nlabels 12 --imb-factor 5 --alg 'supervised'
python toy_examples.py --seed 0 --dataset 'fourspins' --nlabels 12 --imb-factor 5 --alg 'MT'
python toy_examples.py --seed 0 --dataset 'fourspins' --nlabels 12 --imb-factor 5 --alg 'PI'
python toy_examples.py --seed 0 --dataset 'fourspins' --nlabels 12 --imb-factor 5 --alg 'MT' --scl 0.5


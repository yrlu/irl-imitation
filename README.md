# irl-imitation
Implementation of selected Inverse Reinforcement Learning (IRL) algorithms in python/Tensorflow.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6796157.svg)](https://doi.org/10.5281/zenodo.6796157)

```
python demo.py
```

<img src="imgs/cmp.jpg" width="830">

##### Implemented Algorithms

- Linear inverse reinforcement learning (Ng & Russell, 2000)
- Maximum entropy inverse reinforcement learning (Ziebart et al., 2008)
- Maximum entropy deep inverse reinforcement learning (Wulfmeier et al., 2015)

##### Implemented MDPs & Solver

- 2D gridworld
- 1D gridworld
- Value iteration

If you use this software in your publications, please cite it using the following BibTeX entry:

```
@misc{lu2017irl-imitation,
  author = {Lu, Yiren},
  doi = {10.5281/zenodo.6796157},
  month = {7},
  title = {{yrlu/irl-imitation: Implementations of inverse reinforcement learning algorithms in python/Tensorflow}},
  url = {https://github.com/yrlu/irl-imitation},
  year = {2017}
}
```

#### Dependencies

- python 2.7
- cvxopt
- Tensorflow 0.12.1
- matplotlib


#### Linear Inverse Reinforcement Learning

- Following Ng & Russell 2000 paper: [Algorithms for Inverse Reinforcement Learning](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf), algorithm 1

```
$ python linear_irl_gridworld.py --act_random=0.3 --gamma=0.5 --l1=10 --r_max=10
```

<img src="imgs/rmap_gt.jpg" width="200"> <img src="imgs/vmap_gt.jpg" width="200"> <img src="imgs/rmap_lirl.jpg" width="200"> <img src="imgs/rmap_lirl_3d.jpg" width="200"> 

#### Maximum Entropy Inverse Reinforcement Learning

(This implementation is largely influenced by [Matthew Alger's maxent implementation](https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py))

- Following Ziebart et al. 2008 paper: [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
- `$ python maxent_irl_gridworld.py --help` for options descriptions

<!-- ```
$ python maxent_gridworld.py --gamma=0.8 --n_trajs=100 --l_traj=20 --no-rand_start --learning_rate=0.01 --n_iters=20
```

<img src="imgs/rmap_gt_maxent.jpg" width="200"> <img src="imgs/vmap_gt_maxent.jpg" width="200"> <img src="imgs/rmap_maxent.jpg" width="200"> <img src="imgs/rmap_maxent_3d.jpg" width="200"> 
 -->

```
$ python maxent_irl_gridworld.py --height=10 --width=10 --gamma=0.8 --n_trajs=100 --l_traj=50 --no-rand_start --learning_rate=0.01 --n_iters=20
```

<img src="imgs/rmap_gt_maxent_10.jpg" width="200"> <img src="imgs/vmap_gt_maxent_10.jpg" width="200"> <img src="imgs/rmap_maxent_10.jpg" width="200"> <img src="imgs/rmap_maxent_3d_10.jpg" width="200"> 

```
$ python maxent_irl_gridworld.py --gamma=0.8 --n_trajs=400 --l_traj=50 --rand_start --learning_rate=0.01 --n_iters=20
```

<img src="imgs/maxent5_2r.jpg" width="830">

#### Maximum Entropy Deep Inverse Reinforcement Learning

- Following Wulfmeier et al. 2015 paper: [Maximum Entropy Deep Inverse Reinforcement Learning](https://arxiv.org/pdf/1507.04888.pdf). FC version implemented. The implementation does not follow exactly the model proposed in the paper. Some tweaks applied including elu activations, clipping gradients, l2 regularization etc.
- `$ python deep_maxent_irl_gridworld.py --help` for options descriptions

```
$ python deep_maxent_irl_gridworld.py --learning_rate=0.02 --n_trajs=200 --n_iters=20
```

<img src="imgs/deep_maxent_5s.jpg" width="830">

#### MIT License



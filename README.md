# irl-imitation
Implementations of selected inverse reinforcement learning / imitation learning algorithms in Python

#### Algorithms implemented 

- [linear inverse reinforcement learning (Ng & Russell 2000)](#linear-inverse-reinforcement-learning)
- [maximum entropy inverse reinforcement learning (Ziebart et al. 2008)](#maximum-entropy-inverse-reinforcement-learning)

## Linear inverse reinforcement learning

- Following Ng & Russell 2000 paper: [Algorithms for Inverse Reinforcement Learning](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf)

```
$ python linear_irl_gridworld.py --act_random=0.3 --gamma=0.5 --l1=10 --r_max=10
```

<img src="imgs/rmap_gt.jpg" width="200"> <img src="imgs/vmap_gt.jpg" width="200"> <img src="imgs/rmap_lirl.jpg" width="200"> <img src="imgs/rmap_lirl_3d.jpg" width="200"> 

## Maximum entropy inverse reinforcement learning

- Following Ziebart et al. 2008 paper: [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
- `$ python maxent_gridworld.py --help` for options descriptions

<!-- ```
$ python maxent_gridworld.py --gamma=0.8 --n_trajs=100 --l_traj=20 --no-rand_start --learning_rate=0.01 --n_iters=20
```

<img src="imgs/rmap_gt_maxent.jpg" width="200"> <img src="imgs/vmap_gt_maxent.jpg" width="200"> <img src="imgs/rmap_maxent.jpg" width="200"> <img src="imgs/rmap_maxent_3d.jpg" width="200"> 
 -->

```
$ python maxent_gridworld.py --height=10 --width=10 --gamma=0.8 --n_trajs=100 --l_traj=50 --no-rand_start --learning_rate=0.01 --n_iters=20
```

<img src="imgs/rmap_gt_maxent_10.jpg" width="200"> <img src="imgs/vmap_gt_maxent_10.jpg" width="200"> <img src="imgs/rmap_maxent_10.jpg" width="200"> <img src="imgs/rmap_maxent_3d_10.jpg" width="200"> 

```
$ python maxent_gridworld.py --gamma=0.8 --n_trajs=400 --l_traj=50 --rand_start --learning_rate=0.01 --n_iters=20
```

<img src="imgs/maxent5_2r.jpg" width="830">


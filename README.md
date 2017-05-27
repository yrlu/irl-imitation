# irl-imitation
Implementations of selected inverse reinforcement learning / imitation learning algorithms in Python


## Linear Inverse Reinforcement Learning

- Following Ng & Russell 2000 paper: [Algorithms for Inverse Reinforcement Learning](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf)

```
$ python linear_irl_gridworld.py --act_random=0.3 --gamma=0.5 --l1=10 --r_max=10
```

<img src="imgs/rmap_gt.jpg" width="400"> <img src="imgs/vmap_gt.jpg" width="400"> 
<img src="imgs/rmap_lirl.jpg" width="400"> <img src="imgs/rmap_lirl_3d.jpg" width="400"> 
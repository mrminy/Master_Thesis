## Implementation of A3C (Asynchronous Advantage Actor-Critic)

This is a modified version of dennybritz's A3C implementation from https://github.com/dennybritz/reinforcement-learning.

#### Explanation
This version contains a model-based exploration architecture using an autoencoder and a transition prediction model on the latent space. The uncertainty from the model is extracted with MC dropout and bootstrap combined.

#### Requirements
- Gym
- Tensorflow
- Numpy

#### Running

Currently only running in Pycharm works! (change parameters in code if needed)

```
python3 train.py --model_dir /tmp/a3c --env Pong-v0 --t_max 5 --eval_every 300 --parallelism 8 --max_global_steps 20000000
```

See `./train.py --help` for a full list of options. Then, monitor training progress in Tensorboard:

```
tensorboard --logdir=/tmp/a3c
```

#### Components

- [`train.py`](train.py) contains the main method to start training.
- [`estimators.py`](estimators.py) contains the Tensorflow graph definitions for the Policy and Value networks.
- [`worker.py`](worker.py) contains code that runs in each worker threads.
- [`policy_monitor.py`](policy_monitor.py) contains code that evaluates the policy network by running an episode and saving rewards to Tensorboard.

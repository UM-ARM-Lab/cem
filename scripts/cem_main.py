#!/usr/bin/env python
import argparse
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np

from cem.cem_gym import ensure_dir, evaluate_theta, get_elite_indicies
from cem.envs import setup_env
from cem.plotting import plot_history
from link_bot_pycommon import experiments_util


def run_cem(args):
    elite_frac = 0.2
    extra_std = 2.0
    extra_decay_time = 10
    ensure_dir('./{}/'.format(args.env))

    start = time.time()
    num_episodes = args.epochs * args.num_process * args.batch_size
    print('expt of {} total episodes'.format(num_episodes))

    num_elite = int(args.batch_size * elite_frac)
    history = defaultdict(list)

    env, obs_shape, act_shape = setup_env(args.env)
    theta_dim = (obs_shape + 1) * act_shape
    means = np.random.uniform(size=theta_dim)
    stds = np.ones(theta_dim)

    monitor_directory = experiments_util.experiment_name(args.env)

    for epoch in range(args.epochs):
        extra_cov = max(1.0 - epoch / extra_decay_time, 0) * extra_std ** 2

        thetas = np.random.multivariate_normal(
            mean=means,
            cov=np.diag(np.array(stds ** 2) + extra_cov),
            size=args.batch_size
        )

        theta_indeces = range(args.batch_size)
        with Pool(args.num_process) as p:
            rewards = p.starmap(
                partial(evaluate_theta, render=False, monitor=True, epoch=epoch, monitor_directory=monitor_directory,
                        env_name=args.env),
                zip(thetas, theta_indeces))

        rewards = np.array(rewards)

        indicies = get_elite_indicies(num_elite, rewards)
        elites = thetas[indicies]

        means = elites.mean(axis=0)
        stds = elites.std(axis=0)

        history['epoch'].append(epoch)
        history['avg_rew'].append(np.mean(rewards))
        history['std_rew'].append(np.std(rewards))
        history['avg_elites'].append(np.mean(rewards[indicies]))
        history['std_elites'].append(np.std(rewards[indicies]))

        print(
            'epoch {} - {:2.1f} {:2.1f} pop - {:2.1f} {:2.1f} elites'.format(
                epoch,
                history['avg_rew'][-1],
                history['std_rew'][-1],
                history['avg_elites'][-1],
                history['std_elites'][-1]
            )
        )

    end = time.time()
    expt_time = end - start
    print('expt took {:2.1f} seconds'.format(expt_time))

    plot_history(history, args.env, monitor_directory, num_episodes, expt_time)
    num_optimal = 3
    print('epochs done - evaluating {} best thetas'.format(num_optimal))

    best_theta_rewards = [evaluate_theta(theta, i, True, False, None, None, args.env)
                          for i, theta in enumerate(elites[:num_optimal])]
    print('best rewards - {} acoss {} samples'.format(best_theta_rewards, num_optimal))


if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('--num-process', default=2, nargs='?', type=int)
    parser.add_argument('--epochs', default=5, nargs='?', type=int)
    parser.add_argument('--batch-size', default=4096, nargs='?', type=int)
    args = parser.parse_args()

    run_cem(args)

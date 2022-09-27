import agents
import argparse
import envs
import joblib as jl
import time
from datetime import datetime
from typing import Any
from util import io, log


def eval_pk_agent(
    agent_n: int, episodes: int, max_ep_steps: int, seed: int
) -> dict[str, Any]:
    '''
    Evaluation of a single perfect-knowledge (PK) agent.

    Parameters
    ----------
    agent_n : int
        Number of the agent.
    episodes : int
        Number of evaluation episodes.
    max_ep_steps : int
        Maximum number of time steps simulated per each episode.
    seed : int
        RNG seed.

    Returns
    -------
    dict[str, Any]
        Data resulting from the simulation.
    '''
    env = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_ep_steps,
        normalize_observation=False,
        normalize_reward=False
    )
    agents.QuadRotorPKAgent(
        env=env,
        agentname=f'PK_{agent_n}',
        seed=seed
    ).eval(
        env=env,
        n_eval_episodes=episodes,
        deterministic=True,
        seed=seed + 1
    )
    return {'env': env}


def train_lstdq_agent(
    agent_n: int,
    epochs: int,
    train_episodes: int,
    max_ep_steps: int,
    agent_config: dict[str, Any],
    perturbation_decay: float,
    run_name: str,
    seed: int,
    safe: bool
) -> dict[str, Any]:
    '''
    Training of a single LSTD Q learning agent.

    Parameters
    ----------
    agent_n : int
        Number of the agent.
    epochs : int
        Number of training epochs. At the end of each epoch, an RL update
        is carried out.
    train_episodes : int
        Episodes per training epochs.
    max_ep_steps : int
        Maximum number of time steps simulated per each episode.
    agent_config : dict[str, any]
        Agent's configuration.
    perturbation_decay : float
        Decay of exploration perturbations at the end of each epoch.
    run_name : str
        The name of this run.
    seed : int
        RNG seed.
    safe : bool
        Whether to train an unsafe or safe version of the agent.

    Returns
    -------
    dict[str, Any]
        Data resulting from the simulation.
    '''
    logger = None  # log.create_logger(run_name, to_file=False)
    env = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_ep_steps,
        normalize_observation=False,
        normalize_reward=False
    )
    agent = agents.wrappers.RecordLearningData(
        (agents.QuadRotorGPSafeLSTDQAgent
         if safe else
         agents.QuadRotorLSTDQAgent)(
            env=env,
            agentname=f'LSTDQ_{agent_n}',
            agent_config=agent_config,
            seed=seed
        ))
    agent.learn(
        n_epochs=epochs,
        n_episodes=train_episodes,
        seed=seed + 1,
        perturbation_decay=perturbation_decay,
        logger=logger
    )
    return {'env': env, 'agent': agent}


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=50,
                        help='Number of parallel agent to train.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of training episodes per epoch.')
    parser.add_argument('--max_ep_steps', type=int, default=50,
                        help='Maximum number of steps per episode.')
    parser.add_argument('--gamma', type=float, default=0.9792,
                        help='Discount factor.')
    parser.add_argument('--lr', type=float, default=0.3,
                        help='Learning rate.')
    parser.add_argument('--max_perc_update', type=float, default=0.15,
                        help='Maximum percentage update of agent weigths.')
    parser.add_argument('--replay_mem_maxlen_factor', type=int, default=10,
                        help='Replay memory maximum length factor.')
    parser.add_argument('--replay_mem_sample_size', type=float, default=0.7,
                        help='Replay memory sample size (%).')
    parser.add_argument('--perturbation_decay', type=float, default=0.885,
                        help='Exploration perturbance decay.')
    parser.add_argument('--seed', type=int, default=1909, help='RNG seed.')
    parser.add_argument('--eval_pk', action='store_true',
                        help='If passed, evaluates a PK agent.')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Joblib\'s parallel jobs.')
    # only relevant for safe variant of algorithm
    parser.add_argument('--safe', type=bool, default=True,
                        help='If passed, trains the agent\'s safe variant.')
    parser.add_argument('--gp_alpha', type=float, default=1e-10,
                        help='Measurement noise of the GP data.')
    parser.add_argument('--gp_kernel_type', choices=('RBF', 'Matern'),
                        default='RBF', help='Kernel core of GP function.')
    parser.add_argument('--average_violation', type=bool, default=True,
                        help='Reduce GP data by averaging violations.')
    args = parser.parse_args()

    # prepare to launch
    run_name = log.get_run_name()
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start = time.perf_counter()
    agent_config = {
        'gamma': args.gamma,
        'lr': args.lr,
        'max_perc_update': args.max_perc_update,
        'replay_maxlen': args.episodes * args.replay_mem_maxlen_factor,
        'replay_sample_size': args.replay_mem_sample_size,
        'replay_include_last': args.episodes,
        'alpha': args.gp_alpha,
        'kernel_cls': args.gp_kernel_type,
        'average_violation': args.average_violation,
    }
    if args.agents == 1:
        args.n_jobs = 1  # don't parallelize
    if args.safe and (args.n_jobs == -1 or args.n_jobs > 1):
        import os
        os.environ['PYTHONWARNINGS'] = 'ignore'  # ignore warnings
    tot_episodes = args.epochs * args.episodes

    # launch training/evaluation
    print(f'[Simulation {run_name} started at {date}]')
    if args.eval_pk:
        func = lambda n: eval_pk_agent(
            agent_n=n,
            episodes=tot_episodes,
            max_ep_steps=args.max_ep_steps,
            seed=args.seed + (tot_episodes + 1) * n
        )
    else:
        func = lambda n: train_lstdq_agent(
            agent_n=n,
            epochs=args.epochs,
            train_episodes=args.episodes,
            max_ep_steps=args.max_ep_steps,
            agent_config=agent_config,
            perturbation_decay=args.perturbation_decay,
            run_name=run_name,
            safe=args.safe,
            seed=args.seed + (tot_episodes + 1) * n
        )
    with log.tqdm_joblib(desc='Simulation', total=args.agents):
        raw_data = jl.Parallel(n_jobs=args.n_jobs)(
            jl.delayed(func)(i) for i in range(args.agents)
        )

    # save results
    data = {
        f'{key}s': [r[key] for r in raw_data]
        for key in ('env', 'agent') if key in raw_data[0]
    }
    fn = io.save_results(
        filename=run_name,
        date=date,
        args=args,
        simtime=time.perf_counter() - start,
        data=data)

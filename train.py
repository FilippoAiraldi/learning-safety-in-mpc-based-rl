import agents
import argparse
import envs
import joblib as jl
import time
from datetime import datetime
from itertools import count
from typing import Any
from util import io, log
from util.math import NormalizationService


def eval_pk_agent(
    agent_n: int,
    episodes: int,
    max_ep_steps: int,
    seed: int,
    normalized_env: bool
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
    normalized_env : bool
        Whether to train on a normalized version of the environment.

    Returns
    -------
    dict[str, Any]
        Data resulting from the simulation.
    '''
    normalization = NormalizationService() if normalized_env else None
    env = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_ep_steps,
        normalization=normalization
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
    return {'success': True, 'env': env}


def train_lstdq_agent(
    agent_n: int,
    epochs: int,
    train_episodes: int,
    max_ep_steps: int,
    agent_config: dict[str, Any],
    perturbation_decay: float,
    runname: str,
    seed: int,
    normalized_env: bool,
    safe_agent: bool,
    verbose: bool
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
    runname : str
        The name of this run.
    seed : int
        RNG seed.
    normalized_env : bool
        Whether to train on a normalized version of the environment.
    safe_agent : bool
        Whether to train an unsafe or safe version of the agent.
    verbose : bool
        Whether intermediate results should be logged.

    Returns
    -------
    dict[str, Any]
        Data resulting from the simulation.
    '''
    logger = log.create_logger(runname, to_file=False) if verbose else None
    normalization = NormalizationService() if normalized_env else None
    env = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_ep_steps,
        normalization=normalization
    )
    agent = agents.wrappers.RecordLearningData(
        (agents.QuadRotorGPSafeLSTDQAgent
         if safe_agent else
         agents.QuadRotorLSTDQAgent)(
            env=env,
            agentname=f'LSTDQ_{agent_n}',
            agent_config=agent_config,
            seed=seed
        ))
    ok = agent.learn(
        n_epochs=epochs,
        n_episodes=train_episodes,
        seed=seed + 1,
        perturbation_decay=perturbation_decay,
        logger=logger,
        return_info=True
    )[0]
    return {'success': ok, 'env': env, 'agent': agent}


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--runname', type=str, default=None,
                        help='Name of the simulation run.')
    parser.add_argument('--agents', type=int, default=50,
                        help='Number of parallel agent to train.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of training episodes per epoch.')
    parser.add_argument('--max_ep_steps', type=int, default=50,
                        help='Maximum number of steps per episode.')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor.')
    parser.add_argument('--lr', type=float, nargs='+',
                        default=0.03,  # [3e-2, 3e-2, 1e-3, 1e-3, 1e-3],
                        help='Learning rate. Cane be a single float, '
                             'or one per parameter type, or one per '
                             'parameter vector entry.')
    parser.add_argument('--max_perc_update', type=float, default=float('inf'),
                        help='Maximum percentage update of agent weigths.')
    parser.add_argument('--replay_mem_maxlen_factor', type=int, default=1,
                        help='Replay memory maximum length factor.')
    parser.add_argument('--replay_mem_sample_size', type=float, default=1.0,
                        help='Replay memory sample size (%).')
    parser.add_argument('--perturbation_decay', type=float, default=0.885,
                        help='Exploration perturbance decay.')
    parser.add_argument('--seed', type=int, default=1909, help='RNG seed.')
    parser.add_argument('--eval_pk', action='store_true',
                        help='If passed, evaluates a PK agent.')
    parser.add_argument('--normalized', action='store_true',
                        help='Whether to use a normalized variant of env.')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Joblib\'s parallel jobs.')
    parser.add_argument('--verbose', action='store_true')
    # only relevant for safe variant of algorithm
    parser.add_argument('--safe', action='store_true',
                        help='Whether to use a safe variant of agent.')
    parser.add_argument('--gp_alpha', type=float, default=1e-10,
                        help='Measurement noise of the GP data.')
    parser.add_argument('--gp_kernel_type', choices=('RBF', 'Matern'),
                        default='RBF', help='Kernel core of GP function.')
    parser.add_argument('--average_violation', action='store_true',
                        help='Reduce GP data by averaging violations.')
    args = parser.parse_args()

    # prepare to launch
    runname = io.get_runname(candidate=args.runname)
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
    if args.eval_pk:
        func = lambda n: eval_pk_agent(
            agent_n=n,
            episodes=tot_episodes,
            max_ep_steps=args.max_ep_steps,
            normalized_env=args.normalized,
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
            runname=runname,
            normalized_env=args.normalized,
            safe_agent=args.safe,
            seed=args.seed + (tot_episodes + 1) * n,
            verbose=args.verbose
        )
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start = time.perf_counter()

    # launch training/evaluation - perform simulations until the required
    # number of agents has been succesfully simulated
    print(f'[Simulation {runname} started at {date}]')
    raw_data: list[dict[str, Any]] = []
    sim_cnt = count(0)
    agent_cnt = count(0)
    while len(raw_data) < args.agents:
        n_agents = max(args.agents - len(raw_data), 10)
        with log.tqdm_joblib(desc=f'Sim {next(sim_cnt)}', total=n_agents):
            raw_data.extend(
                jl.Parallel(n_jobs=args.n_jobs)(
                    jl.delayed(func)(next(agent_cnt)) for _ in range(n_agents)
                ))
        raw_data = list(filter(lambda o: o['success'], raw_data))

    # save results
    data = {
        f'{key}s': [r[key] for r in raw_data]
        for key in ('env', 'agent') if key in raw_data[0]
    }
    fn = io.save_results(
        filename=runname,
        date=date,
        args=args,
        simtime=time.perf_counter() - start,
        data=data
    )

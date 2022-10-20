import time
from datetime import datetime
from typing import Any
import joblib as jl
import agents
import envs
from util import io, log
from util.math import NormalizationService
from util.configurations import parse_args


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
        normalize_reward=(False,),
        normalization=normalization
    )
    agent = agents.QuadRotorPKAgent(
        env=env,
        agentname=f'PK_{agent_n}',
        seed=seed
    )
    agent.eval(
        env=env,
        n_eval_episodes=episodes,
        deterministic=True,
        seed=seed + 1
    )
    return {'success': True, 'agent': agent}


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
        normalize_reward=(True, agent_config['gamma']),
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
    success = agent.learn(
        n_epochs=epochs,
        n_episodes=train_episodes,
        seed=seed + 1,
        perturbation_decay=perturbation_decay,
        logger=logger,
        return_info=True
    )[0]
    return {'success': success, 'agent': agent}


if __name__ == '__main__':
    # prepare to launch
    args = parse_args()
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
            agent_config={
                'gamma': args.gamma,
                'lr': args.lr,
                'max_perc_update': args.max_perc_update,
                'replay_maxlen': args.episodes * args.replay_mem_maxlen_factor,
                'replay_sample_size': args.replay_mem_sample_size,
                'replay_include_last': args.episodes,
                'alpha': args.gp_alpha,
                'kernel_cls': args.gp_kernel_type,
                'average_violation': args.average_violation,
            },
            perturbation_decay=args.perturbation_decay,
            runname=args.runname,
            normalized_env=args.normalized,
            safe_agent=args.safe,
            seed=args.seed + (tot_episodes + 1) * n,
            verbose=args.verbose
        )
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start = time.perf_counter()

    # launch simulations until the required number of agents has been simulated
    print(f'[Simulation {args.runname.upper()} started at {date}]\n',
          f'Args: {args}')
    data: list[dict[str, Any]] = []
    sim_iter, agent_cnt = 0, 0
    while len(data) < args.agents:
        n_agents = args.agents - len(data)
        with log.tqdm_joblib(desc=f'Simulation {sim_iter}', total=n_agents):
            batch = jl.Parallel(n_jobs=args.n_jobs)(
                jl.delayed(func)(agent_cnt + n) for n in range(n_agents)
            )
        data.extend(filter(lambda o: o['success'], batch))
        sim_iter += 1
        agent_cnt += n_agents

    # save results
    print(f'[Simulated {agent_cnt} agents: {agent_cnt - args.agents} failed]')
    fn = io.save_results(
        filename=args.runname,
        date=date,
        args=args,
        simtime=time.perf_counter() - start,
        agents=[d['agent'] for d in data]
    )

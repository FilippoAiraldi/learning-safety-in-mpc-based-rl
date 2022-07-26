import time
from datetime import datetime
from typing import Any, Optional

import joblib as jl

from agents.quad_rotor_lstdq_agents import (
    QuadRotorGPSafeLSTDQAgent,
    QuadRotorLSTDQAgent,
)
from agents.quad_rotor_pk_agent import QuadRotorPKAgent
from agents.wrappers.record_learning_data import RecordLearningData
from envs.quad_rotor_env import QuadRotorEnv
from util.configurations import parse_args
from util.gp import PriorSafetyKnowledge
from util.io import save_results
from util.log import create_logger, tqdm_joblib


def eval_pk_agent(
    agent_n: int,
    episodes: int,
    max_ep_steps: int,
    seed: int,
) -> dict[str, Any]:
    """
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
    """
    env = QuadRotorEnv.get_wrapped(
        max_episode_steps=max_ep_steps, normalize_reward=(False,)
    )
    agent = QuadRotorPKAgent(env=env, agentname=f"PK_{agent_n}", seed=seed)
    agent.eval(env=env, n_eval_episodes=episodes, deterministic=True, seed=seed + 1)
    return {"success": True, "agent": agent}


def train_lstdq_agent(
    agent_n: int,
    epochs: int,
    train_episodes: int,
    max_ep_steps: int,
    agent_config: dict[str, Any],
    perturbation_decay: float,
    runname: str,
    seed: int,
    safe: bool,
    prior_knowledge: Optional[PriorSafetyKnowledge],
    verbose: bool,
) -> dict[str, Any]:
    """
    Training of a single LSTD Q learning agent.

    Parameters
    ----------
    agent_n : int
        Number of the agent.
    epochs : int
        Number of training epochs. At the end of each epoch, an RL updat is carried out.
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
    safe : bool
        Whether to train an unsafe or safe version of the agent.
    prior_knowledge : PriorSafetyKnowledge
        Dataset of prior knowledge from some previous safe-lstdq simulation.
    verbose : bool
        Whether intermediate results should be logged.

    Returns
    -------
    dict[str, Any]
        Data resulting from the simulation.
    """
    logger = create_logger(runname, to_file=False) if verbose else None
    env = QuadRotorEnv.get_wrapped(
        max_episode_steps=max_ep_steps, normalize_reward=(True, agent_config["gamma"])
    )
    agent = RecordLearningData(
        (QuadRotorGPSafeLSTDQAgent if safe else QuadRotorLSTDQAgent)(
            env=env, agentname=f"LSTDQ_{agent_n}", agent_config=agent_config, seed=seed
        )
    )
    if safe and prior_knowledge is not None:
        agent.unwrapped.gpr_dataset.extend(prior_knowledge.get(size=0.1))
    success = agent.learn(
        n_epochs=epochs,
        n_episodes=train_episodes,
        seed=seed + 1,
        perturbation_decay=perturbation_decay,
        logger=logger,
        return_info=True,
    )[0]
    return {"success": success, "agent": agent}


if __name__ == "__main__":
    # prepare to launch
    args = parse_args()
    tot_episodes = args.epochs * args.episodes
    prior_knowledge = (
        None
        if args.prior is None
        else PriorSafetyKnowledge.from_sim(args.prior, seed=args.seed)
    )
    if args.pk:

        def func(n: int) -> dict[str, Any]:
            return eval_pk_agent(
                agent_n=n,
                episodes=tot_episodes,
                max_ep_steps=args.max_ep_steps,
                seed=args.seed + (tot_episodes + 1) * n,
            )

    else:
        agent_config = {
            "gamma": args.gamma,
            "lr": args.lr,
            "max_perc_update": args.max_perc_update,
            "replay_maxlen": args.episodes * args.replay_mem_size,
            "replay_sample_size": args.replay_mem_sample,
            "replay_include_last": args.episodes,
            "alpha": args.gp_alpha,
            "kernel_cls": args.gp_kernel,
            "average_violation": args.average_violation,
        }

        def func(n: int) -> dict[str, Any]:
            return train_lstdq_agent(
                agent_n=n,
                epochs=args.epochs,
                train_episodes=args.episodes,
                max_ep_steps=args.max_ep_steps,
                agent_config=agent_config,
                perturbation_decay=args.perturbation_decay,
                runname=args.runname,
                safe=args.safe_lstdq,
                prior_knowledge=prior_knowledge,
                seed=args.seed + (tot_episodes + 1) * n,
                verbose=args.verbose,
            )

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = time.perf_counter()

    # launch simulations until the required number of agents has been simulated
    print(f"[Simulation {args.runname.upper()} started at {date}]\n", f"Args: {args}")
    data: list[dict[str, Any]] = []
    sim_iter, agent_cnt = 0, 0
    while len(data) < args.agents:
        n_agents = args.agents - len(data)
        with tqdm_joblib(desc=f"Simulation {sim_iter}", total=n_agents):
            batch = jl.Parallel(n_jobs=args.n_jobs)(
                jl.delayed(func)(agent_cnt + n) for n in range(n_agents)
            )
        data.extend(filter(lambda o: o["success"], batch))
        sim_iter += 1
        agent_cnt += n_agents

    # save results
    print(f"[Simulated {agent_cnt} agents: {agent_cnt - args.agents} failed]")
    fn = save_results(
        filename=args.runname,
        date=date,
        args=args,
        simtime=time.perf_counter() - start,
        agents=[d["agent"] for d in data],
    )

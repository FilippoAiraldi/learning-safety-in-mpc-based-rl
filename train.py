import agents
import argparse
import envs
import joblib as jl
import util
from copy import deepcopy
from typing import Any


def train(
    agent_n: int,
    sessions: int,
    train_episodes: int,
    eval_episodes: int,
    max_ep_steps: int,
    run_name: str,
    seed: int
) -> dict[str, Any]:
    '''
    Training of a single agent.

    Parameters
    ----------
    agent_n : int
        Number of the agent.
    sessions : int
        Number of training sessions. At the end of each session, an RL update
        is carried out.
    train_episodes : int
        Episodes per training sessions.
    eval_episodes : int
        Evaluation episodes at the end of each session.
    max_ep_steps : int
        Maximum number of time steps simulated per each episode.
    run_name : str
        The name of this run.
    seed : int
        RNG seed.

    Returns
    -------
    dict[str, Any]
        Data resulting from the training.
    '''
    # create logger
    logger = util.create_logger(run_name, to_file=True)

    # create envs
    env = envs.QuadRotorEnv.get_wrapped(max_episode_steps=max_ep_steps)
    eval_env = deepcopy(env)

    # create agent and launch training
    #
    # agent = agents.wrappers.RecordLearningData(
    #     agents.QuadRotorLSTDDPGAgent(
    #         env=env,
    #         agentname=f'DPG_{agent_n}',
    #         agent_config={
    #             'replay_maxlen': train_episodes,
    #             'replay_sample_size': train_episodes,
    #         },
    #         seed=seed * (agent_n + 1) * 1000))
    #
    # agent = agents.QuadRotorPIAgent(env=env, agentname=f'PI_{agent_n}')
    #
    agent = agents.wrappers.RecordLearningData(
        agents.LinearLSTDDPGAgent(
            env=env,
            agentname=f'Lin_LSTDDPG_{agent_n}',
            agent_config={
                'replay_maxlen': train_episodes,
                'replay_sample_size': train_episodes,
            },
            seed=seed * (agent_n + 1) * 1000
        ))

    agent.learn(
        n_train_sessions=sessions,
        n_train_episodes=train_episodes,
        eval_env=eval_env,
        n_eval_episodes=eval_episodes,
        seed=seed,
        logger=logger
    )

    return {'env': env, 'eval_env': eval_env, 'agent': agent}


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=1,
                        help='Number of parallel agent to train.')
    parser.add_argument('--sessions', type=int, default=200,
                        help='Number of training sessions.')
    parser.add_argument('--train_episodes', type=int, default=100,
                        help='Number of training episodes per session.')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes per session.')
    parser.add_argument('--max_ep_steps', type=int, default=50,
                        help='Maximum number of steps per episode.')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
    args = parser.parse_args()

    # set up defaults
    util.set_np_mpl_defaults()
    run_name = util.get_run_name()

    # launch training
    const_args = (args.sessions,
                  args.train_episodes,
                  args.eval_episodes,
                  args.max_ep_steps, run_name)
    if args.agents == 1:
        data = [train(0, *const_args, args.seed)]
    else:
        with util.tqdm_joblib(desc='Training', total=args.agents):
            data = jl.Parallel(n_jobs=-1)(
                jl.delayed(train)(
                    i, *const_args, args.seed + i) for i in range(args.agents))

    # save results and launch plotting (is blocking)
    fn = util.save_results(filename=run_name, data=data)
    # import os
    # os.system(f'python visualization.py -fn {fn}')

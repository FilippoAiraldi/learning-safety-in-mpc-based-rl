import agents
import argparse
import envs
import joblib as jl
import numpy as np
import util
from logging import Logger
from typing import Any


# NOTE: if we opt for fixed-duration episodes, then we can better stack
# the recorded data, and perform better updates (instead of looping over
# episodes). However, what about the MPC horizon?


def train(
    agent_n: int,
    sessions: int,
    episodes: int,
    max_ep_steps: int,
    logger: Logger,
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
    episodes : int
        Episodes per training sessions.
    max_ep_steps : int
        Maximum number of time steps simulated per each episode.
    logger : Logger
        A logging utility.
    seed : int
        RNG seed.

    Returns
    -------
    dict[str, Any]
        Data resulting from the training.
    '''
    env = envs.QuadRotorEnv.get_wrapped(max_episode_steps=max_ep_steps)
    agent: agents.QuadRotorDPGAgent = agents.wrappers.RecordLearningData(
        agents.QuadRotorDPGAgent(env=env, agentname=f'DPG_{agent_n}',
                                 agent_config={
                                     'replay_maxlen': episodes,
                                     'replay_sample_size': episodes,
                                 }, seed=seed * (agent_n + 1)))

    # simulate m episodes for each session
    for s in range(sessions):
        # run each episode
        for e in range(episodes):
            # reset env
            state = env.reset(seed=seed * (s * 10 + e))

            # simulate this episode
            for t in range(max_ep_steps):
                # _, _, solution = agent.predict(state, deterministic=True)
                # u, _, _ = agent.predict(state, deterministic=False)
                #
                action, _, sol = agent.predict(
                    state, deterministic=False, perturb_gradient=False)
                #
                assert sol.success, f'{agent_n}|{s}|{e}|{t}: MPC failed.'

                # step environment
                new_state, cost, done, _ = env.step(action)

                # save transition
                agent.save_transition((state, action, cost, new_state), sol)

                # check if episode is done
                if done:
                    break
                state = new_state

            # when the episode is done, consolidate its experience into memory
            agent.consolidate_episode_experience()

        # when all m episodes are done, perform RL update and reduce
        # exploration strength
        agent.update()
        agent.perturbation_strength *= 0.97

        # log session outcomes
        logger.debug(f'{agent_n}|{s}|{e}: '
                     f'J={np.mean(env.cum_rewards[-episodes:]):.3f} '
                     f'||dJ||={agent.update_gradient_norm[-1]:.3e}')

    # return data to be saved
    return {
        'observations': list(env.observations),
        'actions': list(env.actions),
        'rewards': list(env.rewards),
        'cum_rewards': list(env.cum_rewards),
        'episode_lengths': list(env.episode_lengths),
        'exec_times': list(env.exec_times),
        'weight_history': agent.weights_hitory,
        'update_gradient_norm': agent.update_gradient_norm,
    }


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=1,  # 100
                        help='Number of parallel agent to train.')
    parser.add_argument('--sessions', type=int, default=3,  # 20
                        help='Number of training sessions.')
    parser.add_argument('--episodes', type=int, default=3,  # 10
                        help='Number of training episodes per session.')
    parser.add_argument('--max_ep_steps', type=int, default=50,  # 100
                        help='Maximum number of steps per episode.')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
    args = parser.parse_args()

    # set up defaults
    util.set_np_mpl_defaults()
    run_name = util.get_run_name()
    logger = util.create_logger(run_name)

    # launch training
    const_args = (args.sessions, args.episodes, args.max_ep_steps, logger)
    if args.agents == 1:
        data = train(1, *const_args, args.seed)
    else:
        with util.tqdm_joblib(desc='Training', total=args.agents):
            data = jl.Parallel(n_jobs=-1)(
                jl.delayed(train)(
                    i, *const_args, args.seed + i) for i in range(args.agents))

    # save results and launch plotting (is blocking)
    fn = util.save_results(filename=run_name, data=data)
    # import os
    # os.system(f'python visualization.py -fn {fn}')

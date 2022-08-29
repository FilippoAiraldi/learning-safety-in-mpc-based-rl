import agents
import argparse
import envs
import joblib as jl
import numpy as np
import util
from typing import Any


def train(
    agent_n: int,
    sessions: int,
    episodes: int,
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
    episodes : int
        Episodes per training sessions.
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
    logger = util.create_logger(run_name, to_file=False)
    env = envs.QuadRotorEnv.get_wrapped(max_episode_steps=max_ep_steps)
    # agent: agents.QuadRotorLSTDDPGAgent = agents.wrappers.RecordLearningData(
    #     agents.QuadRotorLSTDDPGAgent(env=env, agentname=f'DPG_{agent_n}',
    #                                  agent_config={
    #                                      'replay_maxlen': episodes,
    #                                      'replay_sample_size': episodes,
    #                                  }, seed=seed * (agent_n + 1)))
    #
    agent: agents.TestLSTDDPGAgent = agents.wrappers.RecordLearningData(
        agents.TestLSTDDPGAgent(env=env, agentname=f'DPG_{agent_n}',
                                agent_config={
                                    'replay_maxlen': episodes,
                                    'replay_sample_size': episodes,
                                }, seed=seed * (agent_n + 1)))
    #
    # agent = agents.QuadRotorPIAgent(env=env, agentname=f'PI_{agent_n}')

    # simulate m episodes for each session
    for s in range(sessions):
        # run each episode
        for e in range(episodes):
            # reset env and agent
            state = env.reset(seed=seed * (s * 10 + e))
            agent.reset()

            # simulate this episode
            for t in range(max_ep_steps):
                action_opt, _, sol = agent.predict(state, deterministic=True)
                action, _, _ = agent.predict(state, deterministic=False)
                #
                # action, _, sol = agent.predict(
                #     state, deterministic=False, perturb_gradient=False)
                # action_opt = sol.vals['u_unscaled'][:, 0]
                #
                new_state, r, done, _ = env.step(action)

                # save transition
                if True:
                    agent.save_transition(
                        state, action, action_opt, r, new_state, sol)
                else:
                    logger.warning(
                        f'{agent_n}|{s}|{e}|{t}: MPC failed: {sol.status}.')
                    # The solver can still reach maximum iteration and not
                    # converge to a good solution. If that happens, break the
                    # episode and label the parameters unsafe.
                    raise ValueError('AHHHHHHH')

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
        J_mean = np.mean([env.cum_rewards[i] for i in range(-episodes, 0)])
        logger.debug(f'{agent_n}|{s}|{e}: J_mean={J_mean:.3f} '
                     f'||dJ||={agent.update_gradient_norm[-1]:.3e}; '
                     + agent.weights.values2str())

    # return data to be saved
    return {'env': env, 'agent': agent}


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=1,
                        help='Number of parallel agent to train.')
    parser.add_argument('--sessions', type=int, default=20,
                        help='Number of training sessions.')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of training episodes per session.')
    parser.add_argument('--max_ep_steps', type=int, default=50,
                        help='Maximum number of steps per episode.')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
    args = parser.parse_args()

    # set up defaults
    util.set_np_mpl_defaults()
    run_name = util.get_run_name()

    # launch training
    const_args = (args.sessions, args.episodes, args.max_ep_steps, run_name)
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

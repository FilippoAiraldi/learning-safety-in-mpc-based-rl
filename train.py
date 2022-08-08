import agents
import argparse
import envs
import joblib as jl
import util
from logging import Logger


# NOTE: if we opt for fixed-duration episodes, then we can better stack
# the recorded data, and perform better updates (instead of looping over
# episodes). However, what about the MPC horizon?


def train(episodes: int, max_ep_steps: int, logger: Logger, seed: int) -> dict:
    # initialize env and agent
    env = envs.QuadRotorEnv.get_wrapped(max_episode_steps=max_ep_steps)
    # agent = agents.QuadRotorPIAgent(env=env, agentname='PI', seed=seed * 10)
    agent: agents.QuadRotorDPGAgent = agents.wrappers.RecordLearningData(
        agents.QuadRotorDPGAgent(env=env, agentname='DPG', seed=seed * 69))

    # simulate
    for i in range(1, episodes + 1):
        # reset env
        env.reset(seed=seed * i)

        # simulate this episode
        for t in range(max_ep_steps):
            # _, _, solution = agent.predict(deterministic=True)
            # u, _, _ = agent.predict(deterministic=False)
            #
            u, _, solution = agent.predict(deterministic=False,
                                           perturb_gradient=False)
            #
            assert solution.success, f'Unexpected MPC failure at time {t}.'

            # step environment
            _, r, done, _ = env.step(u)

            # save transition
            agent.save_transition(env.x, u, r, solution)

            # check if episode is done
            if done:
                break

        # perform RL update
        agent.consolidate_episode_experience()
        agent.update()

        # reduce exploration strength
        agent.perturbation_strength *= 0.97

        # log episode outcomes
        logger.debug(f'J={env.cum_rewards[-1]:.3f} - '
                     f'||dJdtheta||={agent.update_gradient_norm[-1]:.3f}')

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
    parser.add_argument('--num_ep', type=int, default=3,  # 30
                        help='Number of training episodes.')
    parser.add_argument('--num_envs', type=int, default=1,  # 100
                        help='Number of parallel environments to train on.')
    parser.add_argument('--max_ep_steps', type=int, default=5,  # 100
                        help='Maximum number of steps per episode.')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed.')
    args = parser.parse_args()

    # set up defaults
    util.set_np_mpl_defaults()
    run_name = util.get_run_name()
    logger = util.create_logger(run_name)

    # launch training
    train_args = (args.num_ep, args.max_ep_steps, logger)
    if args.num_envs == 1:
        data = train(*train_args, 45)
    else:
        with util.tqdm_joblib(desc='Training', total=args.num_envs):
            data = jl.Parallel(n_jobs=-1)(
                jl.delayed(train)(
                    *train_args, 42 + i) for i in range(args.num_envs))

    # save results and launch plotting (is blocking)
    fn = util.save_results(filename=run_name, data=data)
    # import os
    # os.system(f'python visualization.py -fn {fn}')

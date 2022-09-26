import argparse
import numpy as np
import optuna
import os
from agents import (
    QuadRotorLSTDQAgent,
    QuadRotorLSTDDPGAgent,
    QuadRotorGPSafeLSTDQAgent
)
from agents.quad_rotor_base_learning_agent import (
    QuadRotorBaseLearningAgent,
    UpdateError
)
from envs import QuadRotorEnv
from mpc import MPCSolverError
from util import io
from typing import Type, TypeVar


os.environ['PYTHONWARNINGS'] = 'ignore'
AgentType = TypeVar('AgentType', bound=QuadRotorBaseLearningAgent)


def objective(
    trial: optuna.Trial,
    agent_cls: Type[AgentType],
    n_epochs: int,
    n_agents: int,
    max_ep_steps: int,
    seed: int,
) -> float:
    # suggest parameters for this trial
    gamma = trial.suggest_float('gamma', 0.98, 1.0, log=True)
    lr = trial.suggest_float('lr', 1e-3, 5e-1, log=True)
    train_eps = 5  # trial.suggest_categorical(
    # 'train_eps', [1, 5, 10, 20])
    max_perc_update = trial.suggest_categorical(
        'max_perc_update', [0.2, 0.5, 1.0, np.inf])
    replay_maxlen_factor = 10  # trial.suggest_int(
    # 'replay_maxlen_factor', 1, n_epochs // 2)
    replay_mem_sample_size = trial.suggest_float(
        'mem_sample_size', 0.2, 0.8, step=0.1)
    replay_include_last_factor = 1  # trial.suggest_int(
    # 'replay_include_last_factor', 0, 1)
    perturbation_decay = trial.suggest_float(
        'perturbation_decay', 0.5, 1.0, step=0.1)

    # create envs and agents
    agent_config = {
        'gamma': gamma,
        'lr': lr,
        'max_perc_update': max_perc_update,
        'replay_maxlen': train_eps * replay_maxlen_factor,
        'replay_sample_size': replay_mem_sample_size,
        'replay_include_last': train_eps * replay_include_last_factor
    }
    envs: list[QuadRotorEnv] = []
    agents: list[agent_cls] = []
    for n_agent in range(n_agents):
        env = QuadRotorEnv.get_wrapped(
            max_episode_steps=max_ep_steps,
            record_data=False,
            normalize_observation=False,
            normalize_reward=False)

        agent = agent_cls(
            env=env,
            agentname=f'LSTDQ_{n_agent}',
            agent_config=agent_config,
            seed=(seed + n_agent) * (n_agent + 1) * 1000)
        envs.append(env)
        agents.append(agent)

    # train along each epoch, and report average performance after each one
    for n_epoch in range(n_epochs):
        # learn one epoch per agent
        returns = [
            agent.learn_one_epoch(
                n_episodes=train_eps,
                perturbation_decay=perturbation_decay,
                seed=seed + train_eps * n_epoch)
            for agent in agents
        ]

        # average peformance over last n episodes and over all agents
        mean_return = np.mean(returns)

        # report current average performance
        trial.report(mean_return, n_epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # return last epoch's averaged performance as result of the trial
    return mean_return


if __name__ == '__main__':
    # parse fixed parameters of the tuning process
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=5,
                        help='Number of parallel agent to train.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--max_ep_steps', type=int, default=50,
                        help='Maximum number of steps per episode.')
    parser.add_argument('--seed', type=int, default=1909, help='RNG seed.')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of Optuna trials.')
    parser.add_argument('--timeout', type=int, default=6 * 60 * 60,
                        help='Optuna timeout (seconds).')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel Optuna jobs.')
    args = parser.parse_args()

    # create study and begin optimization
    study = optuna.create_study(
        study_name='gp-safe-lstdq-tuning',
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        direction='minimize')
    obj_fn = lambda trial: objective(
        trial,
        agent_cls=QuadRotorGPSafeLSTDQAgent,
        n_epochs=args.epochs,
        n_agents=args.agents,
        max_ep_steps=args.max_ep_steps,
        seed=args.seed)
    study.optimize(
        obj_fn,
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        catch=(MPCSolverError, UpdateError),
        show_progress_bar=False)

    # save results
    io.save_results('tuning/lstdq.pkl', args=args, study=study)

    # display some stats
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print('Study statistics: ')
    print('\tNumber of finished trials: ', len(study.trials))
    print('\tNumber of pruned trials: ', len(pruned_trials))
    print('\tNumber of complete trials: ', len(complete_trials))
    print('Best trial:')
    print('\tValue: ', study.best_trial.value)
    print('\tParams: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')

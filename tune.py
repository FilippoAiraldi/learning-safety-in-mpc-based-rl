import argparse
import numpy as np
import optuna
import os
from agents import \
    QuadRotorLSTDQAgent, QuadRotorGPSafeLSTDQAgent
from agents.quad_rotor_base_learning_agent import \
    QuadRotorBaseLearningAgent, UpdateError
from envs import QuadRotorEnv, NormalizedQuadRotorEnv
from mpc import MPCSolverError
from typing import Any, Type, TypeVar
from util import io


os.environ['PYTHONWARNINGS'] = 'ignore'
EnvType = TypeVar('EnvType', bound=QuadRotorEnv)
AgentType = TypeVar('AgentType', bound=QuadRotorBaseLearningAgent)


def set_fixed_attr(trial: optuna.Trial, name: str, value: Any) -> Any:
    trial.set_user_attr(name, value)
    return value


def objective(
    trial: optuna.Trial,
    env_cls: Type[EnvType],
    agent_cls: Type[AgentType],
    n_epochs: int,
    n_agents: int,
    max_ep_steps: int,
    seed: int,
) -> float:
    # fix parameters of this trial
    train_eps = set_fixed_attr(trial, 'train_eps', 1)
    max_perc_update = set_fixed_attr(trial, 'max_perc_update', float('inf'))
    replay_maxlen_factor = set_fixed_attr(trial, 'replay_maxlen_factor', 1)
    replay_mem_sample_size = set_fixed_attr(trial, 'mem_sample_size', 1.0)
    replay_include_last_factor = set_fixed_attr(
        trial, 'replay_include_last_factor', 1)

    # suggest parameters for this trial
    gamma = trial.suggest_float('gamma', 0.95, 1.0, log=True)
    lr = trial.suggest_float('lr', 1e-6, 5e-1, log=True)
    # train_eps = trial.suggest_categorical('train_eps', [1, 5, 10, 20])
    # max_perc_update = trial.suggest_categorical(
    #     'max_perc_update', [0.2, 0.5, 1.0, np.inf])
    # replay_maxlen_factor = trial.suggest_int(
    #     'replay_maxlen_factor', 1, n_epochs // 2)
    # replay_mem_sample_size = trial.suggest_float(
    #     'mem_sample_size', 0.2, 0.8, step=0.1)
    # replay_include_last_factor = trial.suggest_int(
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
        'replay_include_last': train_eps * replay_include_last_factor,
        # 'alpha': args.gp_alpha,
        # 'kernel_cls': args.gp_kernel_type,
        # 'average_violation': args.average_violation,
    }
    envs: list[env_cls] = []
    agents: list[agent_cls] = []
    for n_agent in range(n_agents):
        env = env_cls.get_wrapped(
            max_episode_steps=max_ep_steps,
            normalize_observation=False,
            normalize_reward=False,
            record_data=False,
            clip_action=False,
            enforce_order=True
        )
        agent = agent_cls(
            env=env,
            agentname=f'Agent_{n_agent}',
            agent_config=agent_config,
            seed=(seed + n_agent) * (n_agent + 1) * 1000
        )
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
    parser.add_argument('--agents', type=int, default=25,
                        help='Number of parallel agent to train.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--max_ep_steps', type=int, default=50,
                        help='Maximum number of steps per episode.')
    parser.add_argument('--normalized', action='store_true',
                        help='Whether to use a normalized variant of env.')
    parser.add_argument('--safe', action='store_true',
                        help='Whether to use a safe variant of agent.')
    parser.add_argument('--seed', type=int, default=1909, help='RNG seed.')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of Optuna trials.')
    parser.add_argument('--timeout', type=int, default=3600 * 3,
                        help='Optuna timeout (seconds).')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel Optuna jobs.')
    args = parser.parse_args()

    # create study and begin optimization
    env_cls = NormalizedQuadRotorEnv if args.normalized else QuadRotorEnv
    agent_cls = QuadRotorGPSafeLSTDQAgent if args.safe else QuadRotorLSTDQAgent
    study = optuna.create_study(
        study_name='tuning',
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        direction='minimize')
    obj_fn = lambda trial: objective(
        trial,
        env_cls=env_cls,
        agent_cls=agent_cls,
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
    io.save_results('tmp.pkl', args=args, study=study)

    # display some stats
    best_trial = study.best_trial
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print('Study statistics: ')
    print('\tNumber of finished trials: ', len(study.trials))
    print('\tNumber of pruned trials: ', len(pruned_trials))
    print('\tNumber of complete trials: ', len(complete_trials))
    print('Best trial:')
    print('\tValue: ', best_trial.value)
    print('\tOptimal params: ')
    for key, value in best_trial.params.items():
        print(f'\t\t{key}: {value}')
    if len(best_trial.user_attrs) > 0:
        print('\tFixed params:')
        for key, value in best_trial.user_attrs.items():
            print(f'\t\t{key}: {value}')

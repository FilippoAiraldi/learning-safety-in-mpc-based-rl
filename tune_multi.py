import numpy as np
import optuna
import util
from agents import QuadRotorLSTDQAgent
from envs import QuadRotorEnv
from mpc import MPCSolverError


def objective(
    trial: optuna.Trial,
    n_epochs: int,
    n_agents: int,
    max_ep_steps: int,
    seed: int,
) -> float:
    # suggest parameters for this trial
    gamma = trial.suggest_float(
        'gamma', 0.98, 1.0, log=True)
    lr = trial.suggest_float(
        'lr', 1e-3, 5e-1, log=True)
    train_eps = trial.suggest_categorical(
        'train_eps', [1, 5, 10, 20])
    max_perc_update = trial.suggest_categorical(
        'max_perc_update', [0.2, 0.5, 1.0, np.inf])
    replay_maxlen_factor = trial.suggest_int(
        'replay_maxlen_factor', 1, n_epochs // 2)
    replay_mem_sample_size = trial.suggest_float(
        'mem_sample_size', 0.2, 0.8, step=0.1)
    replay_include_last_factor = trial.suggest_int(
        'replay_include_last_factor', 0, 1)
    perturbation_decay = trial.suggest_float(
        'perturbation_decay', 0.5, 0.99)

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
    agents: list[QuadRotorLSTDQAgent] = []
    for n_agent in range(n_agents):
        env = QuadRotorEnv.get_wrapped(
            max_episode_steps=max_ep_steps,
            record_data=False,
            normalize_observation=False,
            normalize_reward=False)

        agent = QuadRotorLSTDQAgent(
            env=env,
            agentname=f'LSTDQ_{n_agent}',
            agent_config=agent_config,
            seed=(seed + n_agent) * (n_agent + 1) * 1000)
        envs.append(env)
        agents.append(agent)

    # train along each session, and report average performance after each one
    for n_epoch in range(n_epochs):
        # learn one epoch per agent
        returns = [
            agent.learn_one_epoch(
                n_episodes=train_eps,
                perturbation_decay=perturbation_decay,
                seed=seed + train_eps * n_epoch,
                raises=True)
            for agent in agents
        ]
        # import joblib as jl
        # returns = jl.Parallel(n_jobs=-1)(
        #     jl.delayed(QuadRotorLSTDQAgent.learn_one_epoch)(
        #         agent,
        #         n_episodes=train_eps,
        #         perturbation_decay=perturbation_decay,
        #         seed=seed + train_eps * n_epoch,
        #         raises=True
        #     ) for agent in agents
        # )

        # average peformance over last n episodes and over all agents
        mean_return = np.mean(returns)

        # report current average performance
        trial.report(mean_return, n_epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # return last session's averaged performance as result of the trial
    return mean_return


if __name__ == '__main__':
    # fixed parameters of the tuning process
    pars = {
        'n_epochs': 20,
        'n_agents': 50,
        'max_ep_steps': 50,
        'seed': 42,
        'n_trials': 20,
        'n_jobs': -1
    }

    # create study and begin optimization
    study = optuna.create_study(
        study_name='lstdq-tuning',
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        direction='minimize')
    obj_fn = lambda trial: objective(
        trial,
        n_epochs=pars['n_epochs'],
        n_agents=pars['n_agents'],
        max_ep_steps=pars['max_ep_steps'],
        seed=pars['seed'])
    study.optimize(
        obj_fn,
        n_trials=pars['n_trials'],
        n_jobs=pars['n_jobs'],
        catch=(MPCSolverError,),
        show_progress_bar=False)

    # save results
    util.save_results('tuning/lstdq.pkl', pars=pars, study=study)

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

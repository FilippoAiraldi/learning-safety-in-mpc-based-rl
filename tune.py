import agents
import envs
import numpy as np
import optuna
import util


def learn(
    trial: optuna.Trial,
    env: envs.QuadRotorEnv,
    agent: agents.QuadRotorLSTDQAgent,
    n_train_sessions: int,
    n_train_episodes: int,
    perturbation_decay: float = 0.75,
    seed: int = None,
) -> float:
    # simulate m episodes for each session
    cnt = 0
    for _ in range(n_train_sessions):
        for _ in range(n_train_episodes):
            state = env.reset(seed=None if seed is None else (seed + cnt))
            agent.reset()
            action = agent.predict(state, deterministic=False)[0]
            done, t = False, 0
            tot_reward = 0
            while not done:
                # compute Q(s, a)
                agent.fixed_pars.update({'u0': action})
                solQ = agent.solve_mpc('Q', state)

                # step the system
                state, r, done, _ = env.step(action)
                tot_reward += r

                # compute V(s+)
                action, _, solV = agent.predict(state, deterministic=False)

                # save only successful transitions
                if solQ.success and solV.success:
                    agent.save_transition(r, solQ, solV)
                else:
                    # The solver can still reach maximum iteration and not
                    # converge to a good solution. If that happens, in the
                    # safe variant break the episode and label the
                    # parameters unsafe.
                    raise NotImplementedError()
                t += 1

            # Handle pruning based on the intermediate value.
            print(f'DONE {cnt}: J={tot_reward:.3f}')
            trial.report(tot_reward, cnt)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # when episode is done, consolidate its experience into memory
            agent.consolidate_episode_experience()
            cnt += 1

        # when all m episodes are done, perform RL update and reduce
        # exploration strength
        agent.update()
        agent.perturbation_strength *= perturbation_decay
        agent.perturbation_chance *= perturbation_decay
    return tot_reward


def objective(trial: optuna.Trial):
    # fixed parameters
    sessions = 10
    train_episodes = 5
    max_ep_steps = 50
    seed = np.random.randint(0, int(1e6))

    # suggested parameters
    gamma = trial.suggest_float('gamma', 0.97, 1.0, log=True)
    lr = trial.suggest_float('lr', 1e-3, 1e0, log=True)
    max_perc_update = trial.suggest_categorical(
        'max_perc_update', [1 / 5, 1 / 2, 1, np.inf])
    replay_mem_sample_size = trial.suggest_float(
        'mem_sample_size', 0.1, 1.0, step=0.1)
    perturbation_decay = trial.suggest_float('pert_decay', 0.5, 0.99)

    # create env
    env = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_ep_steps,
        record_data=False,
        normalize_observation=False,
        normalize_reward=False)

    # create agent
    agent_config = {
        'gamma': gamma,
        'lr': lr,
        'max_perc_update': max_perc_update,
        'replay_maxlen': train_episodes * 5,  # fixed
        'replay_sample_size': replay_mem_sample_size,  # [0.1, 1.0]
        'replay_include_last': train_episodes,  # fixed
    }
    agent = agents.QuadRotorLSTDQAgent(
        env=env,
        agentname='LSTDQ',
        agent_config=agent_config,
        seed=seed * 27)

    # train agent
    return learn(trial=trial,
          env=env,
          agent=agent,
          n_train_sessions=sessions,
          n_train_episodes=train_episodes,
          perturbation_decay=perturbation_decay,
          seed=seed)


if __name__ == '__main__':
    study = optuna.create_study(
        study_name='lstdq-tuning',
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        direction='minimize')

    study.optimize(
        objective,
        n_trials=20,
        n_jobs=1,
        catch=(NotImplementedError,))

    util.save_results('tuning/lstdq.pkl', study=study)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

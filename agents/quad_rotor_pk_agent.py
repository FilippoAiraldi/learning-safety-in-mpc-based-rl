from agents.quad_rotor_base_agent import QuadRotorBaseAgent


class QuadRotorPKAgent(QuadRotorBaseAgent):
    '''
    Quad rotor agent with Perfect Knowledge available, i.e., the agent is 
    equipped with the exact values governing the target environment. For this 
    reason, the PK agent is often the baseline controller. Rather obviously, 
    this agent does not implement any RL parameter update paradigm. 
    '''

    def __init__(self, *args, **kwargs) -> None:
        '''
        Initializes a Perfect-Information agent for the quad rotor env.

        Parameters
        ----------
        *args, **kwargs
            See QuadRotorBaseAgent.
        '''
        env_config_dict = kwargs['env'].config.__dict__
        env_pars = ('g', 'thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
                    'roll_d', 'roll_dd', 'roll_gain', 'xf')
        kwargs['fixed_pars'] = {
            # set no random perturbation for this agent
            'perturbation': 0,
            # environment parameters - copied from true values
            # stage cost weights - arbitrary numbers
            **{n: env_config_dict[n] for n in env_pars},
            'w_x': 1e1,  # 8
            'w_u': 1e0,
            'w_s': 1e2,  # 120
            # constraint backoff - arbitrary number
            'backoff': 0.05,
        }

        # initialize class
        super().__init__(*args, **kwargs)

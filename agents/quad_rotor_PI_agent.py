from agents.quad_rotor_base_agent import QuadRotorBaseAgent


class QuadRotorPIAgent(QuadRotorBaseAgent):
    '''
    Quad rotor agent with Perfect Information available, i.e., the agent is 
    equipped with the exact values governing the target environment. For this 
    reason, the PI agent is often the baseline controller. Rather obviously, 
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
        init_pars = {}
        kwargs['init_pars'] = init_pars

        # get the environment configuration - cannot be None
        env_config_dict = kwargs['env'].config.__dict__

        # copy model parameters from env config
        for name in ('g', 'thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
                     'roll_d', 'roll_dd', 'roll_gain', 'xf'):
            init_pars[name] = env_config_dict[name]

        # set others to some arbitrary number - should be tuned
        names_and_vals = [
            ('w_L', 1),
            ('w_V', 1),
            ('w_s', 100),
            ('w_s_f', 100),
            ('backoff', 0.05),
        ]
        for name, val in names_and_vals:
            init_pars[name] = val

        # set no random perturbation for this agent
        kwargs['fixed_pars'] = {'perturbation': 0}

        # initialize class
        super().__init__(*args, **kwargs)

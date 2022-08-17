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
            ('w_Lx', 1e1),
            ('w_Lu', 1e0),
            ('w_Ls', 1e2),
            ('w_Tx', 1e1),
            ('w_Tu', 1e0),
            ('w_Ts', 1e2),
            ('backoff', 0.05),
        ]
        for name, val in names_and_vals:
            init_pars[name] = val

        # set no random perturbation for this agent
        kwargs['fixed_pars'] = {'perturbation': 0}

        # initialize class
        super().__init__(*args, **kwargs)

import numpy as np
from agents.quad_rotor_base_agent import QuadRotorBaseAgent


class QuadRotorPIAgent(QuadRotorBaseAgent):
    '''Quad rotor agent with perfect information available.'''

    def __init__(self, *args, **kwargs) -> None:
        '''
        Initializes a perfect-information agent for the quad rotor env.

        Parameters
        ----------
        *args, **kwargs
            See QuadRotorBaseAgent.
        '''
        # do not instantiate parameters as they will be overwritten
        kwargs['init_pars'] = None
        super().__init__(*args, **kwargs)

        # copy model parameters from env
        for n in ('thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
                  'roll_d', 'roll_dd', 'roll_gain', 'xf'):
            self.weights['value'][n] = getattr(self.env.config, n)

        # set cost weights to some number (arbitrary) - to be tuned
        for n in ('w_L', 'w_V'):
            self.weights['value'][n] = np.ones_like(self.weights['value'][n])
        for n in ('w_s', 'w_s_f'):
            self.weights['value'][n] = \
                100 * np.ones_like(self.weights['value'][n])

        # others (arbitrary) - to be tuned
        self.weights['value']['backoff'] = 0.05

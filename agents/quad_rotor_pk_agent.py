from agents.quad_rotor_base_agents import QuadRotorBaseAgent
from envs.quad_rotor_env import QuadRotorEnv


class QuadRotorPKAgent(QuadRotorBaseAgent):
    """
    Quad rotor agent with Perfect Knowledge available, i.e., the agent is equipped with
    the exact values governing the target environment. For this reason, the PK agent is
    often the baseline controller. Rather obviously, this agent does not implement any
    RL parameter update paradigm.
    """

    def __init__(self, env: QuadRotorEnv, *args, **kwargs) -> None:
        """
        Initializes a Perfect-Information agent for the quad rotor env.

        Parameters
        ----------
        env : QuadRotorEnv
            Environment for which to create the agent.
        *args, **kwargs
            See `agents.QuadRotorBaseAgent`.
        """
        # copy parameters directly from the environment
        env_pars = {
            n: getattr(env.config, n)
            for n in [
                "g",
                "thrust_coeff",
                "pitch_d",
                "pitch_dd",
                "pitch_gain",
                "roll_d",
                "roll_dd",
                "roll_gain",
                "xf",
            ]
        }
        ctrl_pars = {
            "w_x": 1e1,
            "w_u": 1e0,
            "w_s": 1e2,
            "backoff": 0.10,
            "perturbation": 0,  # no random perturbation for this agent
        }
        fixed_pars = env_pars | ctrl_pars
        super().__init__(*args, env=env, fixed_pars=fixed_pars, **kwargs)

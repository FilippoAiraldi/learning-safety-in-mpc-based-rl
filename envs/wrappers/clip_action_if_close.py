import gym
import numpy as np
from gym.error import InvalidAction
from gym.wrappers import ClipAction


class ClipActionIfClose(ClipAction):
    """
    Clips the continuous action within bounds even in spite of small numerical
    violations. If the violations of the bounds is outside tolerace, then an
    error is raised.
    """

    def __init__(self, env: gym.Env, **np_isclose_kwargs) -> None:
        """
        A wrapper for clipping continuous actions within the valid bound only
        if the actions is numerically close to the bounds.

        Parameters
        ----------
        env : gym.Env
            The environment to apply the wrapper
        np_isclose_kwargs : dict
            Optional arguments passed to `numpy.isclose` function to check if
            the action is numerically close to the bounds.
        """
        super().__init__(env)
        self._kw = np_isclose_kwargs

    def action(self, action):
        """
        Clips the action within the valid bounds.

        Parameters
        ----------
        action : array_like
            The action to clip.

        Returns
        -------
        clipped_action : array_like
            The clipped action.

        Raises
        ------
        InvalidAction
            The action is both outside bounds and numerically not close enough
            to them.
        """
        low = self.env.action_space.low
        high = self.env.action_space.high
        if ((action < low) & ~np.isclose(action, low, **self._kw)).any() or (
            (action > high) & ~np.isclose(action, high, **self._kw)
        ).any():
            raise InvalidAction("Action outside bounds' numerical check")
        return super().action(action)

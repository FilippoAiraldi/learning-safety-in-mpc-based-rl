from typing import Generic, TypeVar, Any
import numpy as np
from agents.quad_rotor_base_agents import QuadRotorBaseLearningAgent


AgentType = TypeVar('AgentType', bound=QuadRotorBaseLearningAgent)


class RecordLearningData(Generic[AgentType]):
    def __init__(self, agent: AgentType) -> None:
        '''
        Wraps an agent to allow storing learning-related data.

        Parameters
        ----------
        agent : QuadRotorBaseLearningAgent or subclasses
            The agent instance to wrap.
        '''
        self.agent = agent

        # initialize storages
        self.weights_history: dict[str, list[np.ndarray]] = {
            n: [p.value] for n, p in agent.weights.as_dict.items()
        }
        self.update_gradient: list[np.ndarray] = []

    @property
    def unwrapped(self) -> AgentType:
        '''Returns the unwrapped instance of the agent.'''
        return self.agent

    def learn_one_epoch(
        self, *args, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        returns, grad, weights = self.agent.learn_one_epoch(*args, **kwargs)
        self._save(grad, weights)
        return returns, grad

    def learn(
        self, *args, **kwargs
    ) -> tuple[
        bool,
        np.ndarray,
        list[np.ndarray],
        list[dict[str, np.ndarray]]
    ]:
        ok, returns, grads, weightss = self.agent.learn(*args, **kwargs)
        for grad, weights in zip(grads, weightss):
            self._save(grad, weights)
        return ok, returns, grads, weightss

    def _save(self, grad: np.ndarray, weights: dict[str, np.ndarray]) -> None:
        self.update_gradient.append(grad)
        for n, w in self.weights_history.items():
            w.append(weights[n])

    def __getattr__(self, name: str) -> Any:
        '''Reroutes attributes to the wrapped agent instance.'''
        if name.startswith('_'):
            raise AttributeError(
                f'accessing private attribute \'{name}\' is prohibited.')
        return getattr(self.agent, name)

    def __str__(self) -> str:
        '''Returns the wrapper name and the unwrapped agent string.'''
        return f'<{type(self).__name__}: {self.agent}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the wrapper.'''
        return str(self)

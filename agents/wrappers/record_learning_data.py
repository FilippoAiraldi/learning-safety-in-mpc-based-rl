import numpy as np
from agents.quad_rotor_base_agents import QuadRotorBaseLearningAgent
from typing import Generic, TypeVar, Any


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

        # save additional references from the agent
        for key, value in agent.__dict__.items():
            if key.endswith('_history') or key == '_config':
                attr = ('agent' if key.startswith('_') else 'agent_') + key
                self.__setattr__(attr, value)

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

    def __getattr__(self, name) -> Any:
        '''Reroutes attributes to the wrapped agent instance.'''
        return getattr(self.agent, name)

    def __getstate__(self) -> dict[str, Any]:
        '''Returns the instance's state to be pickled.'''
        state = self.__dict__.copy()
        del state['agent']
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        '''Sets the instance's state after loading from pickle.'''
        self.agent = None
        for attr, val in state.items():
            self.__setattr__(attr, val)

    def __str__(self) -> str:
        '''Returns the wrapper name and the unwrapped agent string.'''
        return f'<{type(self).__name__}: {self.agent}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the wrapper.'''
        return str(self)

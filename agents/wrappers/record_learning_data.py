import numpy as np
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from typing import Generic, TypeVar, Any


AgentType = TypeVar('AgentType', bound=QuadRotorBaseLearningAgent)


class RecordLearningData(Generic[AgentType]):
    def __init__(self, agent: AgentType) -> None:
        '''
        Wraps an agent to allow storing learning-related data. In particular, 
        it wraps the methods 'update' and 'consolidate_episode_experience'.

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
        self.update_gradient_norm: list[np.ndarray] = []

    def update(self, *args, **kwargs) -> np.ndarray:
        grad = self.agent.update(*args, **kwargs)

        # save gradient and its norm
        self.update_gradient.append(grad)
        g = np.linalg.norm(grad, axis=0).squeeze()
        self.update_gradient_norm.append(g.item() if np.isscalar(g) else g)

        # save new weights
        for n, w in self.weights_history.items():
            w.append(self.agent.weights[n].value)
        return grad

    def learn(self, *args, **kwargs) -> Any:
        # trick to pass the self's wrapped instance to the method
        return type(self.agent).learn(self, *args, **kwargs)

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

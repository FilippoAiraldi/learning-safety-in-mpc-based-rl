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
        self._agent = agent

        # initialize storages
        self.weights_hitory: dict[str, list[np.ndarray]] = {
            n: [p.value] for n, p in agent.weights.as_dict.items()
        }
        self.update_gradient_norm: list[np.ndarray] = []

    @property
    def agent(self) -> AgentType:
        return self._agent

    def consolidate_episode_experience(self, *args, **kwargs) -> np.ndarray:
        grad = self._agent.consolidate_episode_experience(*args, **kwargs)
        x = np.linalg.norm(grad, axis=0).squeeze()
        self.update_gradient_norm.append(x.item() if np.isscalar(x) else x)
        return grad

    def update(self, *args, **kwargs) -> None:
        o = self._agent.update(*args, **kwargs)
        for n, w in self.weights_hitory.items():
            w.append(self._agent.weights[n].value)
        return o

    def __getattr__(self, name) -> Any:
        '''Reroutes attributes to the wrapped agent instance.'''
        return getattr(self._agent, name)

    def __str__(self) -> str:
        '''Returns the wrapper name and the unwrapped agent string.'''
        return f'<{type(self).__name__}: {self._agent}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the wrapper.'''
        return str(self)

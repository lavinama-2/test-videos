import numpy as np
from gym import spaces

from rl_agents.agents.common.exploration.abstract import ContinuousDistribution


class OrnsteinUhlenbeck(ContinuousDistribution):
    """
        Adds exploration noise on top of action taken
        Adapted from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(self, action_space, config=None):
        super(OrnsteinUhlenbeck, self).__init__(config)
        self.action_space = action_space
        if isinstance(self.action_space, spaces.Tuple):
            self.action_space = self.action_space.spaces[0]
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("The action space should be continuous")
        self.action_dim = self.action_space.shape[0]
        self._mu = self.config['mu']
        self._theta = self.config['theta']
        self.sigma = self.config['max_sigma']
        self._max_sigma = self.config['max_sigma']
        self._min_sigma = self.config['min_sigma']
        self._decay_period = self.config['decay_period']
        self.values = None
        self.reset()
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(
            mu=0.0,
            theta=0.15,
            max_sigma=0.3,
            min_sigma=0.3,
            decay_period=100_000
        )

    def reset(self):
        self.state = np.ones(self.action_dim) * self._mu
        self.set_time(0)

    def evolve_state(self):
        x = self.state
        dx = self._theta * (self._mu - x) + self.sigma * np.random.randn(
            self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action):
        ou_state = self.evolve_state()
        self.sigma = self._max_sigma - (self._max_sigma - self._min_sigma) \
                     * min(1.0, self._time / self._decay_period)
        return np.clip(action + ou_state,
                       self.action_space.low,
                       self.action_space.high)

    def set_time(self, time):
        self._time = time

    def step_time(self):
        self._time += 1

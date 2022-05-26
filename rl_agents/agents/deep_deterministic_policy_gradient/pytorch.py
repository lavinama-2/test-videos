import torch
from gym import spaces

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.memory import ReplayMemory, Transition
from rl_agents.agents.common.models import size_model_config, model_factory, \
    trainable_parameters
from rl_agents.agents.common.optimizers import loss_function_factory, \
    optimizer_factory
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.utils import choose_device


class DDPGAgent(AbstractAgent):

    def __init__(self, env, config=None):
        super(DDPGAgent, self).__init__(config)
        self.env = env
        self.memory = ReplayMemory(self.config)
        self.exploration_policy = exploration_factory(
            self.config["exploration"], self.env.action_space)
        self.training = True
        self.previous_state = None
        # TODO: rework size_model_config to configure for continuous action spaces
        # size_model_config(self.env, self.config["models"])

        self.actor_net = model_factory(self.config["models"]["actor"])
        self.actor_target_net = model_factory(self.config["models"]["actor"])
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.actor_target_net.eval()

        self.critic_net = model_factory(self.config["models"]["critic"])
        self.critic_target_net = model_factory(self.config["models"]["critic"])
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        self.critic_target_net.eval()

        self.device = choose_device(self.config["device"])
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.actor_target_net.to(self.device)
        self.critic_target_net.to(self.device)
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.actor_optimizer = optimizer_factory(
            self.config["optimizer"]["type"],
            self.actor_net.parameters(),
            **self.config["optimizer"])
        self.critic_optimizer = optimizer_factory(
            self.config["optimizer"]["type"],
            self.critic_net.parameters(),
            **self.config["optimizer"])
        self.steps = 0

    @classmethod
    def default_config(cls):
        return dict(models=dict(
            actor={
                "type": "ActorNetwork",
                "in": 7,
                "base_net": {
                    "type": "MultiLayerPerceptron",
                },
                "out_activation": "TANH",
                "out": 2,
            },
            critic={
                "type": "CriticNetwork",
                "in": 9,
                "base_net": {
                    "type": "MultiLayerPerceptron",
                    "activation": "RELU",
                },
                "out": 1,
            },
        ),
            optimizer=dict(type="ADAM",
                           lr=5e-4,
                           weight_decay=0,
                           k=5),
            loss_function="l2",
            memory_capacity=50000,
            batch_size=100,
            gamma=0.99,
            device="cuda:best",
            exploration=dict(method="OrnsteinUhlenbeck"),
            target_update=1,
            double=True)

    def act(self, state, step_exploration_time=True):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :param step_exploration_time: step the exploration schedule
        :return: an action
        """
        self.previous_state = state
        if step_exploration_time:
            self.exploration_policy.step_time()
        # Handle multi-agent observations
        # TODO: it would be more efficient to forward a batch of states
        if isinstance(state, tuple):
            return tuple(self.act(agent_state, step_exploration_time=False) for
                         agent_state in state)

        # Single-agent setting
        action = self.get_action(state)
        return self.exploration_policy.get_action(action)

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition by performing a Policy Gradient iteration

            - push the transition into memory
            - sample a minibatch
            - compute the bellman residual loss over the minibatch
            - perform one gradient descent step
            - slowly track the policy network with the target network
        :param state: a state
        :param action: an action
        :param reward: a reward
        :param next_state: a next state
        :param done: whether state is terminal
        """
        if not self.training:
            return
        if isinstance(state, tuple) and isinstance(action,
                                                   tuple):  # Multi-agent setting
            [self.memory.push(agent_state, agent_action, reward,
                              agent_next_state, done, info)
             for agent_state, agent_action, agent_next_state in
             zip(state, action, next_state)]
        else:  # Single-agent setting
            self.memory.push(state, action, reward, next_state, done, info)
        batch = self.sample_minibatch()
        if batch:
            policy_loss, critic_loss, _, _ = \
                self.compute_bellman_residual(batch)
            self.step_optimizers(policy_loss, critic_loss)
            self.update_target_networks()

    def sample_minibatch(self):
        if len(self.memory) < self.config["batch_size"]:
            return None
        transitions = self.memory.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    def compute_bellman_residual(self, batch):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(
                tuple(torch.tensor([batch.state], dtype=torch.float))).to(
                self.device)
            action = torch.tensor(batch.action, dtype=torch.float).to(
                self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(
                self.device)
            next_state = torch.cat(
                tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(
                self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(
                self.device)
            batch = Transition(state, action, reward, next_state, terminal,
                               batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.critic_net(batch.state, batch.action)

        with torch.no_grad():
            # Compute V(s_{t+1}) for all next states.
            next_actions = self.actor_target_net(
                batch.next_state).data.cpu().numpy()
            next_state_action_values = \
                self.critic_target_net(batch.next_state,
                                       torch.tensor(next_actions, dtype=torch.float))
            # Compute the expected Q values
            target_state_action_value = batch.reward[:,None] + self.config[
                "gamma"] * next_state_action_values

        # Compute losses
        critic_loss = self.loss_function(state_action_values,
                                         target_state_action_value)
        policy_loss = -self.critic_net(batch.state,
                                       self.actor_net(batch.state)).mean()
        return policy_loss, critic_loss, target_state_action_value, batch

    def step_optimizers(self, policy_loss, critic_loss):
        step_optimizer(self.actor_net, self.actor_optimizer, policy_loss)
        step_optimizer(self.critic_net, self.critic_optimizer, critic_loss)

    def update_target_networks(self):
        # TODO: consider using lossy target updates instead (with tau=0.1)
        self.steps += 1
        if self.steps % self.config["target_update"] == 0:
            self.actor_target_net.load_state_dict(self.actor_net.state_dict())
            self.critic_target_net.load_state_dict(self.critic_net.state_dict())

    def get_action(self, state):
        """
        :param state: s, an environment state
        :return: a=\mu(s) the optimal action of the actor's policy
        """
        return self.get_batch_actions([state])[0]

    def get_batch_actions(self, states):
        """
        Get the optimal actions of several states
        :param states: [s1; ...; sN] an array of states
        :return: values:[a1; ...; aN] the array of optimal actions for each state
        """
        return self.actor_net(torch.tensor(states, dtype=torch.float).to(
            self.device)).data.cpu().numpy()

    def get_state_action_value(self, state, action):
        """
        :param state: s, an environment state
        :param action: a, an action
        :return: Q(s,a) the state action value of the critic's policy
        """
        return self.get_batch_state_action_values([state], [action])[0]

    def get_batch_state_action_values(self, states, actions):
        """
        Get the state-action values of several states
        :param states: [s1; ...; sN] an array of states
        :param actions: [a1; ...; aN] an array of actions
        :return: values:[Q1, ..., QN] the array of state action values for each state action pair
        """
        return self.critic_net(
            torch.tensor(states, dtype=torch.float).to(self.device),
            torch.tensor(actions, dtype=torch.float).to(self.device),
        ).data.cpu().numpy()

    def reset(self):
        pass

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def save(self, filename):
        state = {'actor_dict': self.actor_net.state_dict(),
                 'actor_optimizer': self.actor_optimizer.state_dict(),
                 'critic_dict': self.critic_net.state_dict(),
                 'critic_optimizer': self.critic_optimizer.state_dict()
                 }
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_net.load_state_dict(checkpoint['actor_dict'])
        self.actor_target_net.load_state_dict(checkpoint['actor_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])

        self.critic_net.load_state_dict(checkpoint['critic_dict'])
        self.critic_target_net.load_state_dict(checkpoint['critic_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        return filename

    def set_writer(self, writer):
        super().set_writer(writer)
        obs_shape = self.env.observation_space.shape if isinstance(
            self.env.observation_space, spaces.Box) else \
            self.env.observation_space.spaces[0].shape
        model_input = torch.zeros((1, *obs_shape), dtype=torch.float,
                                  device=self.device)
        self.writer.add_graph(self.actor_net, input_to_model=(model_input,)),
        self.writer.add_scalar("agent/trainable_parameters",
                               trainable_parameters(self.actor_net), 0)

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    def eval(self):
        self.training = False
        self.config['exploration']['method'] = "OrnsteinUhlenbeck"
        self.exploration_policy = exploration_factory(
            self.config["exploration"], self.env.action_space)


def step_optimizer(model_net, optimizer, loss):
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
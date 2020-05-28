import gym
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .agent import Agent
from .replay_buffer import ReplayBuffer
from .world_model import WorldModel
from .disagreement_ensamble import DE


class Dreamer(object):

    def __init__(self, hparams):
        
        self.hparams = hparams

        self.env = gym.make(self.hparams.env)

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.hparams = add_model_specific_args(hparams, self.obs_dim, self.act_dim)

        self.replay_buffer = ReplayBuffer(self.hparams)
        self.world_model = WorldModel(self.hparams, self.obs_dim, self.act_dim)
        self.de = DE(self.hparams, self.obs_dim, self.act_dim)

        # exploration actor critic
        #self.exp_actor_critic = Agent(self.hparams)

        # task actor critic
        #self.task_actor_critic = Agent(self.hparams)

        # trainers
        self.representation_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)
        self.transition_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)
        self.reward_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)
        self.action_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)
        self.value_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)

    def dreamer(self):

        self.initial_random_explore()
        
        while True:

            for step in range(self.update_steps):
                self.learn_dynamics()
                self.learn_behavior()

            self.interact_environment()

    def initial_random_explore(self):
        
        for e in range(self.hparams.num_explore_episodes):

            state = self.env.reset()

            for _ in range(self.hparams.num_explore_steps):

                action = self.env.action_space.sample()

                next_state, reward, terminal, info = self.env.step(action)

                self.replay_buffer.store(state, action, reward, next_state, terminal)

                state = next_state

                if terminal:
                    break
        pass

    def interact_environment(self):
        
        for e in range(self.hparams.num_explore_episodes):

            state = self.env.reset()

            for _ in range(self.hparams.num_explore_steps):

                action = self.exp_actor_critic.act(state)

                next_state, reward, terminal, info = self.env.step(action)

                self.replay_buffer.store(state, action, reward, next_state, terminal)

                state = next_state

                if terminal:
                    break

        pass

    def execute_task_ac(self):
        pass

    def add_episodes(self):
        pass

    def distil_r(self):
        pass

    def fit_world_model(self):

        dl = DataLoader(self.replay_buffer, batch_size=self.hparams.batch_size)
        self.world_model_trainer.fit(self.world_model, train_dataloader=dl)

    def fit_de(self):
        dl = DataLoader(self.replay_buffer, batch_size=self.hparams.batch_size)
        self.de_trainer.fit(self.de, train_dataloader=self.replay_buffer)

    def fit_exp_ac(self):
        dl = DataLoader(self.replay_buffer, batch_size=self.hparams.batch_size)
        self.exp_ac_trainer.fit(self.exp_actor_critic, train_dataloader=self.replay_buffer)

    def fit_task_ac(self):
        dl = DataLoader(self.replay_buffer, batch_size=self.hparams.batch_size)
        self.task_ac_trainer.fit(self.task_actor_critic, train_dataloader=self.replay_buffer)

    @staticmethod
    def add_model_specific_args(parent_parser, obs_sz, act_sz):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--observation_size', default=obs_sz, type=float)
        parser.add_argument('--action_size', default=act_sz, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser

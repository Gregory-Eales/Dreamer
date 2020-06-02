import gym
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from .replay_buffer import ReplayBuffer

from .action_model import ActionModel
from .representation_model import RepresentationModel
from .transition_model import TransitionModel
from .reward_model import RewardModel
from .value_model import ValueModel


class Dreamer(pl.LightningModule):

    def __init__(self, hparams):
        
        self.hparams = hparams

        self.env = gym.make(self.hparams.env)

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape


        #print(self.obs_dim)

        #self.hparams = self.add_model_specific_args(hparams, self.obs_dim, self.act_dim)

        self.replay_buffer = ReplayBuffer(self.hparams)

        # models
        self.representation_model = RepresentationModel(hparams=hparams)
        self.transition_model = TransitionModel(hparams=hparams)
        self.reward_model = RewardModel(hparams=hparams)
        self.action_model = ActionModel(hparams=hparams)
        self.value_model = ValueModel(hparams=hparams)
        
        # trainers
        self.trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)

    def forward(self):
        pass

    def dream(self):

        self.seed_random_episodes()
        
        while True:

            for step in range(self.update_steps):
                self.learn_dynamics()
                self.learn_behavior()

            self.interact_environment()

    def seed_random_episodes(self):
        
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

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}  
            
    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_val_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
    
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_representation = torch.optim.Adam(self.representation_model.parameters(), lr=lr, betas=(b1, b2))
        opt_transition = torch.optim.Adam(self.transition_model.parameters(), lr=lr, betas=(b1, b2))
        opt_reward = torch.optim.Adam(self.reward_model.parameters(), lr=lr, betas=(b1, b2))
        opt_action = torch.optim.Adam(self.action_model.parameters(), lr=lr, betas=(b1, b2))
        opt_value = torch.optim.Adam(self.value_model.parameters(), lr=lr, betas=(b1, b2))
        return [opt_representation, opt_transition, opt_reward, opt_action, opt_value], []

    def train_dataloader(self):
        # REQUIRED
        """
        return DataLoader(MNIST(os.getcwd(), train=True, download=True,
         transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)
        """
        pass

    @staticmethod
    def add_model_specific_args(parent_parser, obs_sz, act_sz):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--observation_size', default=obs_sz, type=tuple)
        parser.add_argument('--action_size', default=act_sz, type=int)

        # training specific (for this model)
        #parser.add_argument('--max_nb_epochs', default=2, type=int)


        return parser


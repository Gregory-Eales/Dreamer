import torch


class RewardModel(torch.nn.Module):


	def __init__(self, in_dim=3, out_dim=1, hparams=None):

		super(RewardModel, self).__init__()

		self.in_dim = in_dim
		self.hid_dim = hid_dim
		self.out_dim = out_dim
		self.num_hid = num_hid

		self.layer = torch.nn.ModuleDict()
		
		self.define_network()

	def define_network(self):

		self.layer["l1"] = torch.nn.Linear(self.in_dim, self.hid_dim)
		
		for i in range(2, self.num_hid+2):
			self.layer["l{}".format(i)] = torch.nn.Linear(self.hid_dim, self.hid_dim)

		self.layer["l{}".format(self.num_hid+2)] = torch.nn.Linear(self.hid_dim, self.out_dim)

		self.leaky_relu = torch.nn.LeakyReLU()

	def forward(self, x):

		out = torch.Tensor(x)
		
		for key in self.layer.keys():
			out = self.layer[key](out)
			out = self.leaky_relu(out)

		return out


	def loss(self):
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
		# REQUIRED
		# can return multiple optimizers and learning_rate schedulers
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

	def train_dataloader(self):
		# REQUIRED
		"""
		return DataLoader(MNIST(os.getcwd(), train=True, download=True,
		 transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)
		"""
		pass

	@staticmethod
	def add_model_specific_args(parent_parser):
		"""
		Specify the hyperparams for this LightningModule
		"""
		# MODEL specific
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--learning_rate', default=0.02, type=float)
		parser.add_argument('--batch_size', default=32, type=int)

		# training specific (for this model)
		parser.add_argument('--max_nb_epochs', default=2, type=int)

		return parser

def main():
	p = ValueNetwork(3, 3, 3, 5)
	p.forward(torch.ones(10, 3))


if __name__ == "__main__":
	main()
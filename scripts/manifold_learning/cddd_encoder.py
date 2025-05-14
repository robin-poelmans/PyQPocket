import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class CDDD_encoder(nn.Module):

	def __init__(self, device):

		super().__init__()
		self.hidden_sizes = [32, 512, 1024, 2048]
		self.vocab_size = 40
		self.latent_size = 512

		# Initialize embedding
		self.char_projection = nn.Linear(self.vocab_size, self.hidden_sizes[0], bias=False)

		# Initialize multi-layer GRU
		self.gru_layers = nn.ModuleList()
		for i in range(3):
			self.gru_layers.append(
				torch.nn.GRU(self.hidden_sizes[i], self.hidden_sizes[i+1], batch_first = True)
				)

		# Initialize latent space
		mlp_in_size = sum(self.hidden_sizes[1:])
		self.dropout = nn.Dropout(p = 0.2)
		self.linear = nn.Linear(mlp_in_size, self.latent_size)
		self.tanh = nn.Tanh()

	def forward(self, x, mask):
		"""
		Parameters
		----------

		x: torch.tensor(batch_size, seq_length, vocab_size)
		mask: torch.tensor(batch_size, seq_length)
			==> Binary: 1 if idx = length of instance

		Returns
		-------

		x_mu: torch.tensor(batch_size, latent_size)
		x_logvar: torch.tensor(batch_size, latent_size)
		"""

		x = self.char_projection(x) # (batch_size, seq_length, hidden_size[0])
		hidden_states_list = []
		mask = mask.unsqueeze(-1)

		for gru_layer in self.gru_layers:
			x, _ = gru_layer(x) # (batch_size, seq_length, hidden_size[i])
			hidden_states_list.append(
				(x * mask).sum(dim = 1) # (batch_size, hidden_size[i])

		x = torch.cat(hidden_states_list, dim = 1) # (batch_size, sum(hidden_size[1:]))
		x = self.dropout(x)

		x_mu = self.tanh(self.linear(x)) # (batch_size, latent_size)

		return x

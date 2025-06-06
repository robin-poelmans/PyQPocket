import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from torchinfo import summary

class VAE_CDDD_encoder(nn.Module):

	def __init__(self):

		super().__init__()
		self.hidden_sizes = [32, 128, 256, 512]
		self.vocab_size = 41
		self.latent_size = 512

		# Initialize embedding
		self.char_projection = nn.Linear(self.vocab_size, self.hidden_sizes[0], bias=False)

		# Initialize multi-layer GRU
		self.gru_layers = nn.ModuleList()
		for i in range(3):
			self.gru_layers.append(
				torch.nn.GRU(self.hidden_sizes[i], self.hidden_sizes[i+1], batch_first = True)
				)

		# Initialize VAE latent space
		mlp_in_size = sum(self.hidden_sizes[1:])
		self.dropout = nn.Dropout(p = 0.2)
		self.batchnorm = nn.BatchNorm1d(mlp_in_size)
		self.fc_mu = nn.Linear(mlp_in_size, self.latent_size)
		self.fc_logvar = nn.Linear(mlp_in_size, self.latent_size)

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
			hidden_states_list.append((x * mask).sum(dim = 1)) # (batch_size, hidden_size[i])

		x = torch.cat(hidden_states_list, dim = 1) # (batch_size, sum(hidden_size[1:]))
		x = self.batchnorm(x)
		x = self.dropout(x)

		x_mu = self.fc_mu(x) # (batch_size, latent_size)
		x_logvar = self.fc_logvar(x) # (batch_size, latent_size)

		return x_mu, x_logvar

class VAE_CDDD_decoder(nn.Module):

	def __init__(self):

		super().__init__()
		self.hidden_sizes = [512, 256, 128, 32]
		self.vocab_size = 41
		self.latent_size = 512

		# Initialize token embedding
		self.token_embedding = nn.Linear(self.vocab_size, self.hidden_sizes[0])

		# Initialize GRU layers from latent embedding
		self.init_state = nn.Linear(self.latent_size, sum(self.hidden_sizes[1:]))

		# Multi-layer GRU
		self.gru_layers = nn.ModuleList()
		for i in range(3):
			self.gru_layers.append(
				nn.GRUCell(self.hidden_sizes[i], self.hidden_sizes[i+1])
			)

		# Output projection layer
		self.projection = nn.Linear(self.hidden_sizes[-1], self.vocab_size)

	def forward(self,
		latent: torch.Tensor,
		target_tokens: Optional[torch.Tensor] = None,
		max_length: int = 350
	) -> Tuple[torch.Tensor, torch.Tensor]:

		"""
		Forward pass for training with teacher forcing.

		Parameters
		----------

		latent: Latent representation [batch_size, embedding_size]
		target_tokens: Target token sequences for teacher forcing [batch_size, seq_len]
		max_length: Maximum sequence length to generate if target_tokens not provided

		Returns
		-------

		outputs: Output logits for each step [batch_size, seq_len, vocab_size]
		predicted: Predicted token IDs [batch_size, seq_len]
		"""

		batch_size = latent.size(0)
		device = latent.device

		if target_tokens is not None:
			seq_len = target_tokens.size(1)
		else:
			seq_len = max_length

		# Initialize hidden state
		combined = self.init_state(latent) # [batch_size, sum(self.hidden_sizes)]

		hidden_states = []
		start_idx = 0
		for size in self.hidden_sizes[1:]:
			end_idx = start_idx + size
			hidden_states.append(combined[:, start_idx:end_idx])
			start_idx = end_idx

		outputs = torch.zeros(batch_size, seq_len, self.vocab_size, device = device)
		predicted = torch.zeros(batch_size, seq_len, device = device)

		sos_token = 1
		input_tokens = torch.full((batch_size,), sos_token, dtype = torch.long, device = device)

		input_one_hot = F.one_hot(input_tokens, num_classes = self.vocab_size).float() # [batch_size,]: one-hot encoding of start token

		# Use teacher forcing
		use_teacher_forcing = (target_tokens is not None)

		# Generate sequence step by step
		for t in range(seq_len):
			x = self.token_embedding(input_one_hot)
			new_hidden_states = []
			for i, gru_cell in enumerate(self.gru_layers):
				h = gru_cell(x, hidden_states[i])
				new_hidden_states.append(h)
				x = h

			# Update logits and sequence prediction
			logits = self.projection(x)
			outputs[:,t,:] = logits
			_, top_idx = torch.topk(logits, 1) # [batch_size, 1]
			predicted[:, t] = top_idx.squeeze(-1)

			# Update before next sequence generation step
			if use_teacher_forcing and t < seq_len - 1:
				input_tokens = target_tokens[:, t+1] # [batch_size,]
			else:
				input_tokens = top_idx.squeeze(-1)
			input_one_hot = F.one_hot(input_tokens, num_classes = self.vocab_size).float() # [batch_size, vocab_size]
			hidden_states = new_hidden_states

		return outputs, predicted

	def compute_loss(self,
		logits: torch.Tensor,
		targets: torch.Tensor,
		target_lengths: torch.Tensor
	) -> torch.Tensor:

		"""
		Compute loss with masked targets

		Parameters
		----------

		logits: Decoder output logits [batch_size, seq_len, vocab_size]
		targets: Target token IDs [batch_size, seq_len]
		target_lengths: Sequence lengths [batch_size,]

		Returns
		-------

		loss: Masked cross-entropy loss
		"""

		batch_size, seq_len, vocab_size = logits.size()
		logits_flat = logits.view(-1, vocab_size) # [batch_size * seq_len, vocab_size]
		targets_flat = targets.view(-1) # [batch_size * seq_len,]

		device = target_lengths.device

		# Create binary matrix from sequence lengths
		ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(device) # [batch_size, seq_len]
		mask = (ids < target_lengths.unsqueeze(1)).float()
		mask_flat = mask.view(-1) # [batch_size * seq_len,]

		# Compute cross-entropy loss
		crossent = F.cross_entropy(logits_flat, targets_flat, reduction = 'none') # [batch_size * seq_len,]
		masked_loss = mask_flat * crossent
		loss = masked_loss.sum() / mask_flat.sum() # Mean reduction over all active entries

		return loss

if __name__ == '__main__':
	model = VAE_CDDD_encoder('cuda:0')
	print(summary(model))
	model = VAE_CDDD_decoder('cuda:0')
	print(summary(model))

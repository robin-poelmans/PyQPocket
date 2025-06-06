import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from scipy.stats import boxcox
from vae_cddd import VAE_CDDD_encoder, VAE_CDDD_decoder
from e3nn_voxel_vae_binary import SE3_Invariant_Encoder, Pocket_decoder
from typing import List, Dict, Tuple
import numpy as np
import random
import os

def setup_distributed(rank, world_size):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'

	# Initialize process group
	dist.init_process_group(
		backend = 'nccl',
		init_method = 'env://',
		world_size = world_size,
		rank = rank)

	torch.cuda.set_device(rank)

def cleanup_distributed():
	dist.destroy_process_group()

class SMILES_tokenizer:
	"""
	Class for the tokenization of SMILES sequences
	"""

	def __init__(self):
		og_dict = {0: '</s>', 1: '#', 2: '%', 3: ')', 4: '(', 5: '+', 6: '-', 7: '1',
			8: '0', 9: '3', 10: '2', 11: '5', 12: '4', 13: '7', 14: '6', 15: '9',
			16: '8', 17: ':', 18: '=', 19: '@', 20: 'C', 21: 'B', 22: 'F', 23: 'I',
			24: 'H', 25: 'O', 26: 'N', 27: 'P', 28: 'S', 29: '[', 30: ']', 31: 'c',
			32: 'i', 33: 'o', 34: 'n', 35: 'p', 36: 's', 37: 'Cl', 38: 'Br', 39: '<s>', 40: '<PAD>'}

		self.vocab_dict = {value: key for key, value in og_dict.items()}

	def tokenize_batch(self, smiles_batch):
		"""
		Parameters
		----------

		smiles_batch: List[str]

		Returns
		-------

		ligand_tokens: torch.tensor(batch_size, seq_length). 
		target_lengths: torch.tensor(batch_size,). dtype = int. Shows length of original sequences before padding, including start and stop token
		mask: torch.tensor(batch_size, seq_length). Binary: 1 if idx = length of instance
		"""

		batch_size = len(smiles_batch)
		vocab_size = len(self.vocab_dict)

		token_list_of_lists = []
		for batch_idx, smiles in enumerate(smiles_batch):
			tokens = []
			i = 0
			while i < len(smiles):
				if i < len(smiles) - 1:
					two_chars = smiles[i:i+2]
					if two_chars in ['Cl', 'Br']:
						tokens.append(self.vocab_dict[two_chars])
						i += 2
						continue
				if smiles[i] in self.vocab_dict:
					tokens.append(self.vocab_dict[smiles[i]])
				i += 1

			tokens = [self.vocab_dict['<s>']] + tokens + [self.vocab_dict['</s>']] # Add start and stop token
			token_list_of_lists.append(tokens)

		target_lengths_list = [len(sublist) for sublist in token_list_of_lists]
		max_seq_length = max(target_lengths_list)
		target_lengths = torch.tensor(target_lengths_list, dtype = torch.float32)

		input_tensor = torch.zeros(batch_size, max_seq_length, dtype = torch.long)
		mask = torch.zeros(batch_size, max_seq_length, dtype = torch.long)

		# Loop over list of lists
		for batch_idx, tokens in enumerate(token_list_of_lists):
			for i in range(max_seq_length):
				if i < len(tokens):
					input_tensor[batch_idx, i] = tokens[i]
				else:
					input_tensor[batch_idx, i] = 40 # Set pad token
			mask[batch_idx, len(tokens) - 1] = 1

		return input_tensor, target_lengths, mask

class ProteinLigandFileDataset(Dataset):
	def __init__(self, protein_names, smiles_list, int_matrix, values, pocket_batch_size):
		"""
		Parameters
		----------

		protein_names: List[str]
		smiles_list: List[str]
		int_matrix: [np.ndarray[n_proteins, n_ligands]]
		values: [np.ndarray[n_proteins, n_features]]
		pocket_batch_size: int - Batch size for pocket encoder
		"""

		self.protein_names = protein_names
		self.smiles_list = smiles_list
		self.int_matrix = int_matrix
		self.values = values
		self.batch_size = pocket_batch_size
		self.tokenizer = SMILES_tokenizer()

		print(f'Number of smiles: {len(smiles_list)}')
		print(f'Int matrix shape: {int_matrix.shape}')

		self.num_batches = len(self.protein_names) // self.batch_size
		if len(self.protein_names) % self.batch_size > 0:
			self.num_batches += 1
		self.pocket_indices = list(range(len(self.protein_names)))

	def __len__(self):
		return self.num_batches

	def _get_batch_ligands(self, pocket_indices: List[int]):
		""" Get all unique ligands interacting with the given proteins.

		Returns
		-------

		index_list [List[int]]:
			List of all ligand entries to load

		w_tensor [torch.tensor(n_prot, n_lig)]:
			binary mask specifying interactions
		"""

		int_submatrix = self.int_matrix[pocket_indices] # Subset rows
		nonzero_mask = np.any(int_submatrix != 0, axis = 0) # [n_ligands,]
		nonzero_indices = list(np.where(nonzero_mask)[0])

		# If more than max cols, randomly select max cols
		max_cols = 2 * self.batch_size
		if len(nonzero_indices) > max_cols:
			selected_indices = random.sample(nonzero_indices, max_cols)
		else:
			selected_indices = nonzero_indices

		w_tensor = torch.tensor(int_submatrix[:, selected_indices], dtype = torch.float32)
		ligand_indices = list(selected_indices)

		return ligand_indices, w_tensor

	def __getitem__(self, batch_idx: int) -> Dict:
		"""
		Get a batch of proteins and their interacting ligands.

		Returns a dictionary with:
			- pocket_tensor: [pok_batch_size, n_channels, grid_size, grid_size, grid_size]
			- pocket_features: [pok_batch_size, n_features]
			- ligand_tensor: [lig_batch_size, seq_length]
			- ligand_length: [lig_batch_size]
			- ligand_mask: [lig_batch_size, seq_length]
			- w_tensor: [pok_batch_size, lig_batch_size]
		"""

		# Sample n random indices for pocket batch
		batch_pocket_indices = random.sample(self.pocket_indices, self.batch_size)
		batch_ligand_indices, w_tensor = self._get_batch_ligands(batch_pocket_indices)

		# Load pocket tensor
		arr_list = []
		for p_idx in batch_pocket_indices:
			protein_name = self.protein_names[p_idx]
			arr = np.load(f'/media/drives/drive3/robin/PYkPocket/training_data/grid_files_volsite/{protein_name}.npy')
			arr[np.isnan(arr)] = 0 # Remove nan values
			temp_tensor = torch.tensor(arr, dtype = torch.float32)
			arr_list.append(temp_tensor)

		pocket_tensor = torch.stack(arr_list)

		# Tokenize ligand SMILES
		batch_ligand_names = [self.smiles_list[i] for i in batch_ligand_indices]
		batch_ligand_tensor, batch_target_lengths, batch_ligand_mask = self.tokenizer.tokenize_batch(batch_ligand_names)

		# Obtain pocket descriptor values
		batch_values = torch.tensor(self.values[batch_pocket_indices], dtype = torch.float32)

		return {
			'pocket_indices': batch_pocket_indices,
			'pocket_tensor': pocket_tensor,
			'pocket_features': batch_values,
			'ligand_tensor': batch_ligand_tensor,
			'ligand_lengths': batch_target_lengths,
			'ligand_mask': batch_ligand_mask,
			'w_tensor': w_tensor}

def create_train_val_split(protein_names: List[str], val_ratio: float = 0.1) -> Tuple[List[str], List[str]]:

	# Randomly select n protein_names for validation
	protein_indices = list(range(len(protein_names)))
	val_num = int(val_ratio * len(protein_indices))
	val_indices = random.sample(protein_indices, val_num)

	train_indices = [x for x in protein_indices if not x in val_indices]

	return train_indices, val_indices

def create_distributed_dataloaders(
	protein_names, smiles_list, int_matrix, values_normed,
	rank, world_size,
	pocket_batch_size = 4,
	num_workers = 4):

	"""
	Create distributed train and validation dataloaders.
	Input:
	Output:
		train_loader, val_loader: Distributed DataLoader objects
	"""

	# Split data at protein level
	train_indices, val_indices = create_train_val_split(
		protein_names = protein_names,
		val_ratio = 0.1)

	train_protein_names = [protein_names[i] for i in train_indices]
	train_int_matrix = int_matrix[train_indices]
	train_values = values_normed[train_indices]

	val_protein_names = [protein_names[i] for i in val_indices]
	val_int_matrix = int_matrix[val_indices]
	val_values = values_normed[val_indices]

	train_dataset = ProteinLigandFileDataset(
		train_protein_names,
		smiles_list,
		train_int_matrix,
		train_values,
		pocket_batch_size)

	val_dataset = ProteinLigandFileDataset(
		val_protein_names,
		smiles_list,
		val_int_matrix,
		val_values,
		pocket_batch_size)

	# Create samplers for distributed training
	train_sampler = DistributedSampler(
		train_dataset,
		num_replicas = world_size,
		rank = rank,
		shuffle = False) # No shuffle at sampler level since dataset handles randomization

	val_sampler = DistributedSampler(
		val_dataset,
		num_replicas = world_size,
		rank = rank,
		shuffle = False)

	# Create dataloaders
	train_loader = DataLoader(
		train_dataset,
		batch_size = 1,
		sampler = train_sampler,
		num_workers = 1,
		pin_memory = False,
		collate_fn = single_item_collate)

	val_loader = DataLoader(
		val_dataset,
		batch_size = 1,
		sampler = val_sampler,
		num_workers = 1,
		pin_memory = False,
		collate_fn = single_item_collate)

	return train_loader, val_loader, train_sampler, val_sampler

def single_item_collate(batch):
	return batch[0]

def run_epoch_distributed(pocket_encoder, pocket_decoder, ligand_encoder, ligand_decoder, dataloader, optimizer, device, sampler = None, rank = 0, epoch = 0, training = True):
	if training:
		pocket_encoder.train()
		ligand_encoder.train()
		pocket_decoder.train()
		ligand_decoder.train()
	else:
		pocket_encoder.eval()
		ligand_encoder.eval()
		pocket_decoder.eval()
		ligand_decoder.eval()

	if sampler is not None:
		sampler.set_epoch(0)

	epoch_loss = 0.0
	num_batches = 0

	context_manager = torch.no_grad() if not training else torch.enable_grad()

	pbar = tqdm(dataloader, disable = not (rank == 0))

	# KL annealing
	start_step = 2000
	end_step = 50000
	steepness = 2.0

	with context_manager:
		for batch in pbar:
			if training:
				optimizer.zero_grad()

			# Load data to device
			pocket_tensor = batch['pocket_tensor'].to(device)
			features_real = batch['pocket_features'].to(device)
			ligand_tensor = batch['ligand_tensor'].to(device)
			ligand_lengths = batch['ligand_lengths'].to(device)
			ligand_mask = batch['ligand_mask'].to(device)
			w_tensor = batch['w_tensor'].to(device)

			# Compute embeddings
			pocket_mu, pocket_logvar = pocket_encoder(pocket_tensor)
			pocket_epsilon = torch.randn_like(pocket_logvar).to(device)
			X_init = pocket_mu + torch.exp(0.5 * pocket_logvar) * pocket_epsilon

			ligand_one_hot = F.one_hot(ligand_tensor, num_classes = 41).float()
			ligand_mu, ligand_logvar = ligand_encoder(ligand_one_hot, mask = ligand_mask)
			ligand_epsilon = torch.randn_like(ligand_logvar).to(device)
			Y_init = ligand_mu + torch.exp(0.5 * ligand_logvar) * ligand_epsilon

			# Aggregation of latent spaces
			X_num = pocket_encoder.module.delta[0] * X_init + pocket_encoder.module.delta[1]*(w_tensor@Y_init) # [pok_batch_size, lat]
			X_norm = pocket_encoder.module.delta[0] * torch.ones(X_init.shape[0], device = device) + pocket_encoder.module.delta[1]*(w_tensor@torch.ones(Y_init.shape[0], device = device)) # [pok_batch_size]
			pocket_embeddings = X_num / X_norm.unsqueeze(1)

			Y_num = pocket_encoder.module.delta[1] * Y_init + pocket_encoder.module.delta[0]*(w_tensor.t()@X_init) # [lig_batch_size, lat]
			Y_norm = pocket_encoder.module.delta[1] * torch.ones(Y_init.shape[0], device = device) + pocket_encoder.module.delta[0]*(w_tensor.t()@torch.ones(X_init.shape[0], device = device)) # [lig_batch_size]
			ligand_embeddings = Y_num / Y_norm.unsqueeze(1)

			x_loss = torch.norm(pocket_embeddings - X_init) / (X_init.shape[0] * X_init.shape[1])
			y_loss = torch.norm(ligand_embeddings - Y_init) / (Y_init.shape[0] * Y_init.shape[1])
			combo_loss = x_loss + y_loss

			# Manifold alignment loss
			ma_loss = torch.norm(pocket_embeddings - w_tensor@ligand_embeddings) / (X_init.shape[0] * X_init.shape[1])

			# Ligand reconstruction loss
			logits, _ = ligand_decoder(ligand_embeddings, ligand_tensor)
			lig_loss = ligand_decoder.module.compute_loss(logits, ligand_tensor, ligand_lengths)

			# Pocket reconstruction loss
			features_pred = pocket_decoder(pocket_embeddings) # [pocket_batch_size, num_features]
			pok_loss = F.mse_loss(features_pred, features_real)

			recon_loss = lig_loss + pok_loss

			# KL divergence
			kl_pocket = -0.5 * torch.mean(1 + pocket_logvar - pocket_mu.pow(2) - pocket_logvar.exp())
			kl_ligand = -0.5 * torch.mean(1 + ligand_logvar - ligand_mu.pow(2) - ligand_logvar.exp())
			kl_loss = kl_pocket + kl_ligand

			# KL annealing
			current_step = epoch*520 + num_batches
			normalized = (current_step - start_step) / (end_step - start_step)
			x = normalized*2*steepness - steepness
			kappa = 1.0 / (1.0 + np.exp(-x))

			# Total loss is weighted sum
			ALPHA = 200
			BETA = 100
			GAMMA = 0.3
			total_loss = ALPHA * combo_loss + BETA * ma_loss + GAMMA*recon_loss + kappa*kl_loss

			if training:
				total_loss.backward()
				optimizer.step()

			pbar.set_postfix({'total': f'{total_loss.item():.4f}', 'combo': f'{ALPHA*combo_loss.item():.4f}', 'ma': f'{BETA*ma_loss.item():.4f}', 'recon': f'{GAMMA*recon_loss.item():.4f}', 'kl': f'{kappa*kl_loss.item():.4f}'})
			epoch_loss += total_loss.item()
			num_batches += 1

	avg_loss = torch.tensor([epoch_loss / num_batches], device = device)
	dist.all_reduce(avg_loss, op = dist.ReduceOp.SUM)
	avg_loss = avg_loss.item() / dist.get_world_size()

	return avg_loss

def train_distributed(rank, world_size, protein_names, smiles_list, int_matrix, values_mean, values_std, values_normed):
	# Setup distributed environment
	setup_distributed(rank, world_size)

	print('Initializing models')
	# Initialize models
	pocket_encoder = SE3_Invariant_Encoder()

	num_features = values_normed.shape[1]
	values_mean = torch.tensor(values_mean)
	values_std = torch.tensor(values_std)
	pocket_decoder = Pocket_decoder(num_features)
	pocket_decoder.set_target_stats(values_mean, values_std)

	ligand_encoder = VAE_CDDD_encoder()
	ligand_decoder = VAE_CDDD_decoder()

	device = torch.device(f'cuda:{rank}')

	# Move models to current device
	pocket_encoder = pocket_encoder.to(device)
	pocket_decoder = pocket_decoder.to(device)
	ligand_encoder = ligand_encoder.to(device)
	ligand_decoder = ligand_decoder.to(device)

	# Wrap models with DDP
	pocket_encoder = DDP(pocket_encoder, device_ids = [rank])
	pocket_decoder = DDP(pocket_decoder, device_ids = [rank])
	ligand_encoder = DDP(ligand_encoder, device_ids = [rank])
	ligand_decoder = DDP(ligand_decoder, device_ids = [rank])

	optimizer = torch.optim.Adam(
		list(pocket_encoder.parameters()) + list(pocket_decoder.parameters()) + list(ligand_encoder.parameters()) + list(ligand_decoder.parameters()),
		lr = 1e-4)

	print('Creating dataloaders')

	train_loader, val_loader, train_sampler, val_sampler = create_distributed_dataloaders(
		protein_names,
		smiles_list,
		int_matrix,
		values_normed,
		rank,
		world_size,
		pocket_batch_size = 12,
		num_workers = 1)

	# Training loop
	num_epochs = 150
	best_val_loss = float('inf')
	patience = 10
	patience_counter = 0

	train_loss_list = []
	val_loss_list = []

	print('Starting training')
	for epoch in range(num_epochs):

		# KL annealing
		x = (epoch / num_epochs)*2 - 1

		train_loss = run_epoch_distributed(
			pocket_encoder = pocket_encoder,
			pocket_decoder = pocket_decoder,
			ligand_encoder = ligand_encoder,
			ligand_decoder = ligand_decoder,
			dataloader = train_loader,
			optimizer = optimizer,
			device = device,
			sampler = train_sampler,
			rank = rank,
			epoch = epoch,
			training = True)

		val_loss = run_epoch_distributed(
			pocket_encoder = pocket_encoder,
			pocket_decoder = pocket_decoder,
			ligand_encoder = ligand_encoder,
			ligand_decoder = ligand_decoder,
			dataloader = val_loader,
			optimizer = optimizer,
			device = device,
			sampler = val_sampler,
			rank = rank,
			epoch = epoch,
			training = False)

		# Print metrics on rank 0: (in principle, this should be the same on all devices, DDP ensures complete synchronization)
		if rank == 0:
			print(f"Epoch {epoch+1}/{num_epochs}")
			print(f"Train Loss: {train_loss:.4f}")
			print(f"Val Loss: {val_loss:.4f}")

			train_loss_list.append(train_loss)
			val_loss_list.append(val_loss)

			# Early stopping
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
			else:
				patience_counter += 1
				if patience_counter >= patience:
					print(f"Early stopping triggered after {epoch+1} epochs")
					break

		# Synchronize processes after each epoch:
		dist.barrier()

	if rank == 0:
		print('Training complete!')

		torch.save({
			'pocket_encoder': pocket_encoder.module.state_dict(), # Working with DDP: save unwrapped model
			'ligand_encoder': ligand_encoder.module.state_dict(),
			'pocket_decoder': pocket_decoder.module.state_dict(),
			'ligand_decoder': ligand_decoder.module.state_dict(),
			'optimizer': optimizer.state_dict()
		}, 'vae_model_volsite.pt')

		# Write train_loss and val_loss to output
		with open('loss_values.txt', 'w') as file:
			file.write('train_loss,val_loss \n')
			for train_loss, val_loss in zip(train_loss_list, val_loss_list):
				file.write(f'{train_loss:.4f},{val_loss:.4f} \n')

	# Clean up distributed resources
	cleanup_distributed()

def main():
	# Load train data
	loaded = np.load('wormhole_train_data.npz')
	int_matrix = loaded['int_matrix']
	int_matrix = np.delete(int_matrix, 1141, axis = 0) # Weird sample, delete
	pocket_names_arr = loaded['pocket_names']
	pocket_names = [str(x) for x in pocket_names_arr]
	del pocket_names[1141]
	smiles_arr = loaded['smiles']
	smiles_list = [str(x) for x in smiles_arr]

	print(f'Int matrix shape: {int_matrix.shape}')
	print(f'Smiles list length: {len(smiles_list)}')

	# Load pocket descriptors
	descriptors_df = pd.read_csv('volsite_descriptors_v3.csv')
	for col in ['cloud_volume', 'Min_CoM_to_surf']:
		descriptors_df[col] = descriptors_df[col] + 1e-8
		descriptors_df[col], _ = boxcox(descriptors_df[col])

	value_columns = ['max_box_edge', 'min_box_edge', 'box_edge_ratio', 'cloud_volume', 'sphericity', 'polar_surface_proportion', 'surface_volume_ratio',
			'Min_CoM_to_surf', 'Max_CoM_to_surf', 'min_max_com_angle','num_ph4_points'] # Ignore filename

	init_values = descriptors_df[value_columns].to_numpy()
	values_mean = np.mean(init_values, axis = 0)
	values_std = np.std(init_values, axis = 0)
	values_normed = (init_values - values_mean) / values_std

	# Launch distributed training:
	world_size = 4

	import torch.multiprocessing as mp
	mp.spawn(
		train_distributed,
		args = (world_size, pocket_names, smiles_list, int_matrix, values_mean, values_std, values_normed),
		nprocs = world_size)

if __name__ == '__main__':
	main()

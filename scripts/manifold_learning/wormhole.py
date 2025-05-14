import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from cddd_encoder import CDDD_encoder
from pocket_encoder import SE3_Invariant_Encoder
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
		target_lengths = torch.FloatTensor(target_lengths_list)

		input_tensor = torch.zeros(batch_size, max_seq_length)
		mask = torch.zeros(batch_size, max_seq_length)

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
	def __init__(self, protein_names, ligand_names, w_total, protein_dir, pocket_batch_size):
		"""
		Parameters
		----------

		protein_names: List[str]
		ligand_names: List[str]
		w_total: np.ndarray [len(protein_names) + len(ligand_names), len(protein_names) + len(ligand_names)]
		protein_dir: str - Path to .npz files
		pocket_batch_size: int - Batch size for pocket encoder
		"""

		self.protein_names = protein_names
		self.ligand_names = ligand_names
		self.w_total = w_total
		self.protein_dir = protein_dir
		self.batch_size = pocket_batch_size
		self.tokenizer = SMILES_tokenizer()

		self.protein_to_ligands = {}
		with open('interaction_mapper.txt', 'r') as file:
			for line in file:
				line = line.strip()
				key_part, value_part = line.split(':', 1)
				key = int(key_part)
				value_str = value_part.strip()[1:-1]
				if value_str:
					# Convert each element to integer
					values = [int(x.strip()) for x in value_str.split(',')]
				else:
					values = []
				self.protein_to_ligands[key] = values

		self.num_batches = len(self.protein_names) // self.batch_size
		if len(self.protein_names) % self.batch_size > 0:
			self.num_batches += 1
		self.pocket_indices = list(range(len(self.protein_names)))

	def __len__(self):
		return self.num_batches

	def _get_batch_ligands(self, pocket_indices: List[int]) -> Tuple[List[int], np.ndarray]:
		""" Get all unique ligands interacting with the given proteins."""
		base_ligands = set()
		additional_ligands = set()
		for p_idx in pocket_indices:
			interacting_ligands = self.protein_to_ligands[p_idx]
			if len(interacting_ligands) == 1:
				base_ligands.add(interacting_ligands[0])
			elif len(interacting_ligands) > 1:
				random_sample = random.sample(interacting_ligands, 1)[0]
				base_ligands.add(random_sample)
				new_ligands = [x for x in interacting_ligands if x != random_sample]
				additional_ligands.update(new_ligands)

		additional_ligands_list = [x for x in additional_ligands if not x in base_ligands]
		if len(additional_ligands_list) > self.batch_size:
			sample_indices = random.sample(additional_ligands_list, self.batch_size)
		else:
			sample_indices = additional_ligands_list

		prot_len = 42358
		final_sample_indices = list(base_ligands) + sample_indices
		ligand_indices = [x + prot_len for x in final_sample_indices]
		full_indices = pocket_indices + ligand_indices

		w_submatrix = self.w_total[np.ix_(full_indices, full_indices)]
		degrees = np.sum(w_submatrix, axis = 1)
		diag_submatrix = np.diag(degrees)
		laplacian = diag_submatrix - w_submatrix

		# Norm laplacian to L_hat
		degrees_sqrt = np.sqrt(degrees)
		diag_sqrt = np.diag(1.0 / degrees_sqrt)
		L_hat = diag_sqrt @ laplacian @ diag_sqrt

		return final_sample_indices, L_hat

	def __getitem__(self, batch_idx: int) -> Dict:
		"""
		Get a batch of proteins and their interacting ligands.

		Returns a dictionary with:
			- protein_features: [pok_batch_size, n_channels, grid_size, grid_size, grid_size]
			- protein_ids: list of protein indices
			- ligand_tokens: [lig_batch_size, seq_length, num_tokens]
			- ligand_mask: [lig_batch_size, seq_length]
			- ligand_ids: list of ligand indices
			- laplacian: joint laplacian of batch
		"""

		# Sample n random indices for pocket batch
		batch_pocket_indices = random.sample(self.pocket_indices, self.batch_size)
 		batch_ligand_indices, laplacian_submatrix = self._get_batch_ligands(batch_pocket_indices)

		# Load pocket tensor
		arr_list = []
		for p_idx in batch_pocket_indices:
			protein_name = self.protein_names[p_idx]
			loaded = np.load(f'/media/drives/drive3/robin/PYkPocket/training_data/grid_files_1A_res/{protein_name}.npz')
			arr = loaded.f.arr_0
			temp_tensor = torch.tensor(arr, dtype = torch.float32)
			arr_list.append(temp_tensor)

		pocket_tensor = torch.stack(arr_list)

		# Tokenize ligand SMILES
		batch_ligand_names = [self.ligand_names[i] for i in batch_ligand_indices]
		batch_ligand_tensor, batch_target_lengths, batch_ligand_mask = self.tokenizer.tokenize_batch(batch_ligand_names)

		laplacian_tensor = torch.tensor(laplacian_submatrix, dtype = torch.float32)

		return {
			'pocket_tensor': pocket_tensor,
			'ligand_tensor': batch_ligand_tensor,
			'ligand_lengths': batch_target_lengths,
			'ligand_mask': batch_ligand_mask,
			'laplacian_tensor': laplacian_tensor}

def create_train_val_split(protein_names: List[str], val_ratio: float = 0.1) -> Tuple[List[str], List[str]]:

	# Randomly select n protein_names for validation
	protein_indices = list(range(len(protein_names)))
	val_num = int(val_ratio * len(protein_indices))
	val_indices = random.sample(protein_indices, val_num)

	train_names = []
	val_names = []

	for i in protein_indices:
		if i in val_indices:
			val_names.append(protein_names[i])
		else:
			train_names.append(protein_names[i])

	return train_names, val_names

def create_distributed_dataloaders(
	protein_names, ligand_names, w_total,
	protein_dir, rank, world_size,
	pocket_batch_size = 4,
	num_workers = 4):

	"""
	Create distributed train and validation dataloaders.
	Input:
	Output:
		train_loader, val_loader: Distributed DataLoader objects
	"""

	# Split data at protein level
	train_protein_names, val_protein_names = create_train_val_split(
		protein_names = protein_names,
		val_ratio = 0.1)

	train_dataset = ProteinLigandFileDataset(
		train_protein_names,
		ligand_names,
		w_total,
		protein_dir,
		pocket_batch_size)

	val_dataset = ProteinLigandFileDataset(
		val_protein_names,
		ligand_names,
		w_total,
		protein_dir,
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
		num_workers = num_workers,
		pin_memory = True,
		collate_fn = single_item_collate)

	val_loader = DataLoader(
		val_dataset,
		batch_size = 1,
		sampler = val_sampler,
		num_workers = num_workers,
		pin_memory = True,
		collate_fn = single_item_collate)

	return train_loader, val_loader, train_sampler, val_sampler

def single_item_collate(batch):
	return batch[0]

def train_epoch_distributed(pocket_encoder, ligand_encoder, dataloader, optimizer, device, sampler = None):
	pocket_encoder.train()
	ligand_encoder.train()

	if sampler is not None:
		sampler.set_epoch(0)

	epoch_loss = 0.0
	num_batches = 0

	for batch in tqdm(dataloader, desc = f'Training on GPU {device}'):

		optimizer.zero_grad()
		# Load data to device
		pocket_tensor = batch['pocket_tensor'].to(device)
		ligand_tensor = batch['ligand_tensor'].to(device)
		ligand_lengths = batch['ligand_lengths'].to(device)
		ligand_mask = batch['ligand_mask'].to(device)
		L_hat = batch['laplacian_tensor'].to(device)

		# Compute embeddings
		pocket_embeddings = pocket_encoder(pocket_tensor)
		ligand_embeddings = ligand_encoder(ligand_tensor)

		F = torch.cat((pocket_embeddings, ligand_embeddings), 0) # Concatenate pocket and ligand embeddings

		# Project to nearest orthogonal matrix
		u, s, v = torch.svd(F, some=True)
		F_hat = u@v.t()

		# Compute manifold alignment loss
		loss = torch.trace(F_hat.t()@L_hat@F_hat) / 512

		# Zero gradients, perform a backward pass and update the weights
		F_hat.retain_grad()
		optimizer.zero_grad()

		# Scale loss and gradients
		loss.backward(retain_graph = True)

		# Project Euclidean gradient onto tangent space of Stiefel Manifold (to get Riemannian gradient)
		skew = 0.5 * (F_hat.t()@F_hat.grad - F_hat.grad.t()@F_hat)
		term1 = F_hat@skew
		eye_matrix = torch.eye(len(proj_outputs), device=device)
		term2 = (eye_matrix - F_hat@F_hat.t())@F_hat
		rgrad = term1 + term2

		# Second backward pass, now with the Riemannian gradient
		optimizer.zero_grad()
		F_hat.backward(rgrad)
		optimizer.step() # DDP accumulates all gradients before doing the optimization!

		epoch_loss += loss.item()
		num_batches += 1

	avg_loss = torch.tensor([epoch_loss / num_batches], device = device)
	dist.all_reduce(avg_loss, op = dist.ReduceOp.SUM)
	avg_loss = avg_loss.item() / dist.get_world_size()

	return avg_loss

def validate_epoch_distributed(pocket_encoder, ligand_encoder, dataloader, device, sampler = None):
	pocket_encoder.eval()
	ligand_encoder.eval()

	if sampler is not None:
		sampler.set_epoch(0)

	epoch_loss = 0.0
	num_batches = 0

	with torch.no_grad():
		for batch in tqdm(dataloader, desc = f'Training on GPU {device}'):

			optimizer.zero_grad()
			# Load data to device
			pocket_tensor = batch['pocket_tensor'].to(device)
			ligand_tensor = batch['ligand_tensor'].to(device)
			ligand_lengths = batch['ligand_lengths'].to(device)
			ligand_mask = batch['ligand_mask'].to(device)
			L_hat = batch['laplacian_tensor'].to(device)

			# Compute embeddings
			pocket_embeddings = pocket_encoder(pocket_tensor)
			ligand_embeddings = ligand_encoder(ligand_tensor)

			F = torch.cat((pocket_embeddings, ligand_embeddings), 0) # Concatenate pocket and ligand embeddings

			# Project to nearest orthogonal matrix
			u, s, v = torch.svd(F, some=True)
			F_hat = u@v.t()

			# Compute manifold alignment loss
			loss = torch.trace(F_hat.t()@L_hat@F_hat) / 512

                epoch_loss += loss.item()
                num_batches += 1

	avg_loss = torch.tensor([epoch_loss / num_batches], device = device)
	dist.all_reduce(avg_loss, op = dist.ReduceOp.SUM)
	avg_loss = avg_loss.item() / dist.get_world_size()

	return avg_loss

def train_distributed(rank, world_size, protein_names, ligand_names, w_total, protein_dir):
	# Setup distributed environment
	setup_distributed(rank, world_size)

	print('Initializing models')
	# Initialize models
	pocket_encoder = SE3_Invariant_Encoder()
	ligand_encoder = CDDD_encoder()

	device = torch.device(f'cuda:{rank}')

	# Move models to current device
	pocket_encoder = pocket_encoder.to(device)
	ligand_encoder = ligand_encoder.to(device)

	# Wrap models with DDP
	pocket_encoder = DDP(pocket_encoder, device_ids = [rank])
	ligand_encoder = DDP(ligand_encoder, device_ids = [rank])

	optimizer = torch.optim.Adam(
		list(pocket_encoder.parameters()) + list(ligand_encoder.parameters()),
		lr = 1e-4)

	print('Creating dataloaders')

	train_loader, val_loader, train_sampler, val_sampler = create_distributed_dataloaders(
		protein_names,
		ligand_names,
		w_total,
		protein_dir,
		rank,
		world_size,
		pocket_batch_size = 64,
		num_workers = 1)

	# Training loop
	num_epochs = 100
	best_val_loss = float('inf')
	patience = 10 # Set patience value for early stopping
	patience_counter = 0

	train_loss_list = []
	val_loss_list = []

	print('Starting training')
	for epoch in range(num_epochs):
		train_loss = train_epoch_distributed(
			pocket_encoder = pocket_encoder,
			ligand_encoder = ligand_encoder,
			dataloader = train_loader,
			optimizer = optimizer,
			device = device,
			sampler = train_sampler)

		val_loss = validate_epoch_distributed(
			pocket_encoder = pocket_encoder,
			ligand_encoder = ligand_encoder,
			dataloader = val_loader,
			device = device,
			sampler = val_sampler)

		# Print metrics on rank 0: (in principle, this should be the same on all devices, DDP ensures complete synchronization)
		if rank == 0:
			print(f"Epoch {epoch+1}/{num_epochs}")
			print(f"Train Loss: {train_loss:.4f}")
			print(f"Val Loss: {val_loss:.4f}")

			train_loss_list.append(train_loss)
			val_loss_list.append(val_loss)

			if (epoch + 1) % 5 == 0:
				torch.save({
					'pocket_encoder': pocket_encoder.module.state_dict(),
					'ligand_encoder': ligand_encoder.module.state_dict(),
					'optimizer': optimizer.state_dict()
					}, f'wormhole_checkpoint_{epoch}.pt')

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
			'pocket_encoder': pocket_encoder.module.state_dict(),
			'ligand_encoder': ligand_encoder.module.state_dict(),
			'optimizer': optimizer.state_dict()
		}, 'final_model_distributed.pt')

		# Write train_loss and val_loss to output
		with open('loss_values.txt', 'w') as file:
			file.write('train_loss,val_loss \n')
			for train_loss, val_loss in zip(train_loss_list, val_loss_list):
				file.write(f'{train_loss:.4f},{val_loss:.4f} \n')

	# Clean up distributed resources
	cleanup_distributed()

def main():
	loaded = np.load('train_data_v3.npz')
	row_titles = loaded['prot_names']
	protein_list = [str(x) for x in row_titles]
	col_titles = loaded['smiles_names']
	ligand_list = [str(x) for x in col_titles]
	protein_dir = '/media/drives/drive3/robin/PYkPocket/training_data/grid_files_1A_res'
	w_total = loaded['w_total'].astype(np.float16)

	# Launch distributed training:
	world_size = torch.cuda.device_count()

	import torch.multiprocessing as mp
	mp.spawn(
		train_distributed,
		args = (world_size, protein_list, ligand_list, w_total, protein_dir),
		nprocs = world_size)

if __name__ == '__main__':
	main()

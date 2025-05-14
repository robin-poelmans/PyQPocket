import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import BatchNorm, Gate, Dropout

class ConvolutionVoxel(nn.Module):

	def __init__(self, irreps_in, irreps_out, kernel_size, cutoff = True, **kwargs):
		super().__init__()

		self.irreps_in = o3.Irreps(irreps_in)
		self.irreps_out = o3.Irreps(irreps_out)
		self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2, p = 1) # SO3-equivariant, no mirroring since proteins are chiral
		self.kernel_size = kernel_size
		self.padding = self.kernel_size // 2

		# Self-connection:
		self.sc = Linear(self.irreps_in, self.irreps_out)

		# Connection with neighbors
		r = torch.linspace(-1, 1, self.kernel_size)
		lattice = torch.stack(torch.meshgrid(r, r, r, indexing = 'ij'), dim = -1) # [r, r, r, R^3]
		emb = soft_one_hot_linspace(
			x = lattice.norm(dim = -1),
			start = 0.0,
			end = 1.0,
			number = self.kernel_size,
			basis = 'smooth_finite',
			cutoff = True)

		self.register_buffer('emb', emb)

		sh = o3.spherical_harmonics(self.irreps_sh, lattice, True, "component")  # [r, r, r, irreps_sh.dim]
		self.register_buffer("sh", sh)

		self.tp = FullyConnectedTensorProduct(
			self.irreps_in,
			self.irreps_sh,
			self.irreps_out,
			shared_weights = False,
			compile_left_right = False,
			compile_right = True)

		self.weight = nn.Parameter(torch.randn(self.kernel_size, self.tp.weight_numel))

	def forward(self, x):

		sc = self.sc(x.transpose(1, 4)).transpose(1, 4) # Linear connection going from irreps_in to irreps_out
		weight_prod = (self.emb @ self.weight)
		weight_prod = weight_prod / (self.kernel_size ** (3/2))
		kernel = self.tp.right(self.sh, weight_prod)
		kernel = torch.einsum('xyzio->oixyz', kernel)
		conv_result = nn.functional.conv3d(x, kernel, padding=self.padding)
		return 0.2 * sc + 0.8 * conv_result

class LowPassFilter(nn.Module):
	def __init__(self, scale, stride = 2):
		super().__init__()

		sigma = 0.5 * (scale ** 2 - 1)**0.5
		size = int(1 + 2 * 2.5 * sigma)
		if size % 2 == 0:
			size += 1

		r = torch.linspace(-1, 1, size)
		lattice = torch.stack(torch.meshgrid(r, r, r, indexing = 'ij'), dim = -1)
		lattice = (size // 2) * lattice

		kernel = torch.exp(-lattice.norm(dim = -1).pow(2) / (2 * sigma**2))
		kernel = kernel / kernel.sum()
		kernel = kernel[None, None]
		self.register_buffer('kernel', kernel)

		self.scale = scale
		self.stride = stride
		self.size = size

	def forward(self, x):
		out = x.reshape(-1, 1, *x.shape[-3:])
		out = nn.functional.conv3d(out, self.kernel, padding = self.size // 2, stride = self.stride)
		out = out.reshape(*x.shape[:-3], *out.shape[-3:])
		return out

class ConvolutionBlock(nn.Module):
	def __init__(self, irreps_in, irreps_out, kernel_size, dropout_prob):
		super().__init__()

		irreps_scalars = o3.Irreps( [ (mul, ir) for mul, ir in irreps_out if ir.l == 0 ] )
		irreps_gated   = o3.Irreps( [ (mul, ir) for mul, ir in irreps_out if ir.l > 0  ] )
		fe = sum(mul for mul,ir in irreps_gated if ir.p == 1)
		irreps_gates = o3.Irreps(f"{fe}x0e").simplify()
		activation_gate = [torch.sigmoid]
		activation_scalar = [torch.relu]

		if irreps_gates.dim == 0:
			irreps_gates = irreps_gates.simplify()
			activation_gate = []

		self.gate = Gate(irreps_scalars, activation_scalar, irreps_gates, activation_gate, irreps_gated)
		self.conv = ConvolutionVoxel(irreps_in, self.gate.irreps_in, kernel_size)
		self.batchnorm = BatchNorm(self.gate.irreps_in)
		self.dropout = Dropout(irreps_out, dropout_prob) # Only takes effect during training, during evaluation simply returns x

	def forward(self, x):

		x = self.conv(x)
		x = self.batchnorm(x.transpose(1, 4))
		x = self.gate(x)
		x = self.dropout(x).transpose(1, 4)
		return x

class SE3_Invariant_Encoder(nn.Module):
	def __init__(self):
		super().__init__()

		dropout_prob = 0.05
		irreps_list = [
			'19x0e',
			'32x0e + 24x1e + 16x2e',
			'64x0e + 48x1e + 32x2e',
			'128x0e + 96x1e + 64x2e',
			'256x0e']

		kernel_sizes = [5, 5, 3, 3]

		self.blocks = nn.ModuleList()
		for i in range(len(irreps_list) - 1):
			irreps_in = o3.Irreps(irreps_list[i])
			irreps_out = o3.Irreps(irreps_list[i+1])
			self.blocks.append(ConvolutionBlock(irreps_in, irreps_out, kernel_sizes[i], dropout_prob))

		self.pool = LowPassFilter(2.0)

		self.linear = nn.Linear(256, 512)
		self.tanh = nn.Tanh()

	def forward(self, x):
		for i, block in enumerate(self.blocks):
			x = block(x)
			x = self.nan_guard(x)
			if i < len(self.blocks) - 1:
				x = self.pool(x)

		x = x.mean(dim = (2,3,4)) # Average spatial pooling
		x = self.tanh(self.linear(x))
		return x # [batch_size, 512]

# Machine Learning for Times Series
# DILATE Project
#
# Ben Kabongo & Martin Brosset
# M2 MVA


import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F


class DilateLoss(nn.Module):

    def __init__(self, alpha: float, gamma: float, device: torch.device):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def compute_Delta_i(outputs_i, targets_i):
        return 
    

    def forward(self, outputs, targets):
        batch_size, length = outputs.size(0), outputs.size(1)

        L_shape = 0
        L_temporal = 0

        Delta = torch.zeros((batch_size, length, length))
        for i in range(batch_size):
            Delta[i] = ((targets[i] ** 2).sum(1).view(-1, 1) + 
                        (outputs[i] ** 2).sum(1).view(1, -1) -
                        2.0 * torch.mm(targets[i], torch.transpose(outputs[i], 0, 1)))

		# TODO:

        L_dilate = self.alpha * L_shape + (1 - self.alpha) * L_temporal
        return L_dilate, L_shape, L_temporal
    

# Soft-DTW: a Differentiable Loss Function for Time-Series
# http://proceedings.mlr.press/v70/cuturi17a/cuturi17a.pdf

INF = 1e9

def soft_DTW_forward(Delta, gamma):
	def min_gamma(a):
		if gamma == 0:
			return np.min(a)
		z = -a/gamma
		max_z = np.max(z)
		return -gamma * (max_z + np.log(np.sum(np.exp(z - max_z))))
	
	N = len(Delta)
	R = INF * np.ones((2 + N, 2 + N))
	R[0, 0] = 0
	for j in range(1, 1 + N):
		for i in range(1, 1 + N):
			R[i, j] = Delta[i -1, j -1] + min_gamma([R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]])
	return R


def soft_DTW_backward(Delta, R, gamma):
	N = len(Delta)
	D = np.zeros((N + 2, N + 2))
	D[1: 1 + N, 1: 1 + N] = Delta
	E = np.zeros((N + 2, N + 2))
	E[-1, -1] = 1
	R[:, -1] = -INF
	R[-1, :] = -INF
	R[-1, -1] = R[-2, -2]
	for j in range(N, 0, -1):
		for i in range(N, 0, -1):
			a = np.exp((R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma)
			b = np.exp((R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma)
			c = np.exp((R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma)
			E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
	return E[1: 1 + N, 1: 1 + N]
	

class SoftDTWFunction(Function):

	@staticmethod
	def forward(ctx, Delta, gamma):
		dtw = 0
		batch_size, output_lenght = Delta.size(0), Delta.size(1)
		R = torch.zeros((batch_size, 2 + output_lenght, 2 + output_lenght)).to(Delta.device)
		for i in range(batch_size):
			Ri = soft_DTW_forward(Delta[i].detach().cpu().numpy(), gamma)
			dtw += Ri[-2, -2]
			R[i] = torch.FloatTensor(Ri).to(Delta.device)
		ctx.save_for_backward(Delta, R, torch.FloatTensor([gamma]).to(Delta.device))
		return dtw / batch_size
  
	@staticmethod
	def backward(ctx, grad_output):
		Delta, R, gamma = ctx.saved_tensors
		gamma = gamma.item()
		batch_size, output_lenght = Delta.size(0), Delta.size(1)
		E = torch.zeros((batch_size, output_lenght, output_lenght)).to(grad_output.device) 
		for i in range(batch_size):    
			Ei = soft_DTW_backward(Delta[i].detach().cpu().numpy(), R[i].detach().cpu().numpy(), gamma)
			E[i] = torch.FloatTensor(Ei).to(grad_output.device)
		return grad_output * E, None

# Machine Learning for Times Series
# DILATE Project
#
# Ben Kabongo & Martin Brosset
# M2 MVA

# DILATE Loss


import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
    

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
			R[i, j] = Delta[i -1, j -1] + min_gamma(np.array([R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]]))
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


## Code from : https://github.com/vincent-leguen/DILATE/blob/master/loss/path_soft_dtw.py
## TODO : write ours

def my_max(x, gamma):
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


def my_min(x,gamma) :
    min_x, argmax_x = my_max(-x, gamma)
    return - min_x, argmax_x


def my_max_hessian_product(p, z, gamma):
    return  ( p * z - p * np.sum(p * z) ) /gamma


def my_min_hessian_product(p, z, gamma):
    return - my_max_hessian_product(p, z, gamma)


def dtw_grad(theta, gamma):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            v, Q[i, j] = my_min(np.array([V[i, j - 1],
                                                V[i - 1, j - 1],
                                                V[i - 1, j]]) , gamma)                        
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m,0,-1):
        for j in range(n,0,-1):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                      Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                      Q[i + 1, j, 2] * E[i + 1, j]
    
    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


def dtw_hessian_prod(theta, Z, Q, E, gamma):
    m = Z.shape[0]
    n = Z.shape[1]

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            V_dot[i, j] = Z[i - 1, j - 1] + \
                          Q[i, j, 0] * V_dot[i, j - 1] + \
                          Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                          Q[i, j, 2] * V_dot[i - 1, j]

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)
    E_dot = np.zeros((m + 2, n + 2))

    for j in range(n,0,-1):
        for i in range(m,0,-1):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                          Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                          Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                          Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                          Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                          Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]


class PathDTWFunction(Function):
    @staticmethod
    def forward(ctx, D, gamma):
        batch_size,N,N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)
        
        grad_gpu = torch.zeros((batch_size, N ,N)).to(device)
        Q_gpu = torch.zeros((batch_size, N+2 ,N+2,3)).to(device)
        E_gpu = torch.zeros((batch_size, N+2 ,N+2)).to(device)  
        
        for k in range(0,batch_size): # loop over all D in the batch    
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k,:,:], gamma)     
            grad_gpu[k,:,:] = torch.FloatTensor(grad_cpu_k).to(device)
            Q_gpu[k,:,:,:] = torch.FloatTensor(Q_cpu_k).to(device)
            E_gpu[k,:,:] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu,D, Q_gpu ,E_gpu, gamma_gpu) 
        return torch.mean(grad_gpu, dim=0) 
    
    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()
        
        batch_size,N,N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N ,N)).to(device)
        for k in range(0,batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k,:,:], Z, Q_cpu[k,:,:,:], E_cpu[k,:,:], gamma)
            Hessian[k:k+1,:,:] = torch.FloatTensor(hess_k).to(device)

        return  Hessian, None


## DILATE Loss


class DilateLoss(nn.Module):

	def __init__(self, alpha: float, gamma: float, device: torch.device):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.device = device
		self.soft_DTW = SoftDTWFunction.apply
		self.path_DTW = PathDTWFunction.apply

	def forward(self, targets, outputs):
		batch_size, length = outputs.size(0), outputs.size(1)

		Delta = torch.zeros((batch_size, length, length))
		for i in range(batch_size):
			Delta[i] = ((targets[i] ** 2).sum(1).view(-1, 1) + 
                        (outputs[i] ** 2).sum(1).view(1, -1) -
                        2.0 * torch.mm(targets[i], torch.transpose(outputs[i], 0, 1)))
		L_shape = self.soft_DTW(Delta, self.gamma)

		Omega = ((torch.range(0, length - 1).view(-1, 1) - (torch.range(0, length - 1))) ** 2) / (length ** 2)

		# TODO:
		A = self.path_DTW(Delta, self.gamma)
		L_temporal = (A * Omega).sum()

		L_dilate = self.alpha * L_shape + (1 - self.alpha) * L_temporal
		return L_dilate, L_shape, L_temporal
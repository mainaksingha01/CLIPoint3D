import torch 
from torch import nn
import torch.nn.functional as F

class EntropyLoss(nn.Module):
	def __init__(self):
		super().__init__()
	
	def forward(self, target_prob):
		full_enp = torch.zeros(target_prob.shape[0])
		target_prob = F.normalize(target_prob, dim=0)
		
		for i in range(len(target_prob)):
			total_en = 0
			for j in range(target_prob.shape[1]):
				total_en = total_en - target_prob[i][j] * torch.log(target_prob[i][j] + 1e-8)
			full_enp[i] = total_en
		avg_full_enp = torch.mean(full_enp)
		return avg_full_enp

	
class Wasserstein1Loss(nn.Module):
    """
    Implementation of Wasserstein-1 (Earth Mover's) distance loss.
    Assumes inputs are probability distributions (sum to 1).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Ensure inputs are proper probability distributions
        pred = pred / torch.sum(pred, dim=-1, keepdim=True)
        target = target / torch.sum(target, dim=-1, keepdim=True)
        
        # Compute cumulative distribution functions
        pred_cdf = torch.cumsum(pred, dim=-1)
        target_cdf = torch.cumsum(target, dim=-1)
        
        # Compute Wasserstein-1 distance
        wasserstein_distance = torch.sum(torch.abs(pred_cdf - target_cdf), dim=-1)
        
        return wasserstein_distance.mean()
	

import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        """
        Initialize Contrastive Loss module
        
        Args:
            margin (float): Margin value for dissimilar pairs. Default is 1.0
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate contrastive loss
        
        Args:
            output1 (torch.Tensor): First output tensor [batch_size, feature_dim]
            output2 (torch.Tensor): Second output tensor [batch_size, feature_dim]
            label (torch.Tensor): Labels tensor [batch_size], 1 for similar, 0 for dissimilar
            
        Returns:
            torch.Tensor: Computed contrastive loss
        """
        # Calculate Euclidean distance between outputs
        euclidean_distance = F.pairwise_distance(
            output1, 
            output2,
            p=2, # Euclidean distance
        )
        
        if not label:
            N = output1.size(0)
            label = torch.tensor([1] * (N//2) + [0] * (N//2)).to(output1.device)
        
        # Calculate loss for similar pairs (label = 1)
        similar_loss = label * torch.pow(euclidean_distance, 2)
        
        # Calculate loss for dissimilar pairs (label = 0)
        dissimilar_loss = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        # Combine losses and take mean
        loss = torch.mean(similar_loss + dissimilar_loss) / 2.0
        
        return loss

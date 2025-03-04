from typing import List
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    Text Encoder Module.
    - Loads a pretrained model (via AutoModel) and its tokenizer.
    - Tokenizes a list of text strings.
    - Returns the [CLS] token embedding as a sentence-level representation.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.text_hidden_dim = self.text_encoder.config.hidden_size

    def forward(self, text_list: str | List | List[List[str]]) -> torch.Tensor:
        """
        Args:
            text_list: list of strings (e.g. cell type labels)
        Returns:
            cls_embedding: Tensor of shape (batch, text_hidden_dim)
        """
        encoded = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True) # Tokenize input text (cell type)
        device = next(self.text_encoder.parameters()).device    # For accelerator compatibility
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.text_encoder(**encoded)  # Generate text embeddings
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token embedding (first token) for each sequence
        return cls_embedding


class CrossModalAttention(nn.Module):
    """
    A PyTorch module that applies cross-attention between two input features using Multihead Attention.
    Each input feature is attended to by the other input feature (replacing q and k/v with each other).

    Given two input feature tensors with different dimensions (dim1 and dim2), this module applies:
    - Attention from input1 (query) attending to input2 (key, value).
    - Attention from input2 (query) attending to input1 (key, value).
    
    Parameters:
    -----------
    dim1 : int
        Embedding dimension of input1.
    dim2 : int
        Embedding dimension of input2.
    num_heads : int
        Number of attention heads.
    
    Inputs:
    -------
    input1 : torch.Tensor
        Tensor of shape (batch_size, dim1).
    input2 : torch.Tensor
        Tensor of shape (batch_size, dim2).

    Returns:
    --------
    output1 : torch.Tensor
        Cross-attended representation of input1 (batch_size, dim1).
    output2 : torch.Tensor
        Cross-attended representation of input2 (batch_size, dim2).
    """

    def __init__(self, 
                hidden_dim: int,
                dim1: int,
                dim2: int,
                num_heads: int = 8,
                dropout: float = 0.1,
                internal_attn = False,
    ):
        super().__init__()
        self.x_attn = nn.MultiheadAttention(
            embed_dim=1 if internal_attn else hidden_dim,
            num_heads=1 if internal_attn else num_heads,
            batch_first=True
        )
        self.proj_1 = nn.Linear(dim1, hidden_dim)
        self.proj_2 = nn.Linear(dim2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.internal_attn = internal_attn


    def forward(self, input1: torch.Tensor, input2: torch.Tensor, return_attn: bool = False):
        
        input1 = self.dropout(self.proj_1(input1))
        input2 = self.dropout(self.proj_2(input2))
        
        # Add sequence length dimension (seq_len=1)
        input1 = input1.unsqueeze(1)   # Shape: (batch_size, 1, dim1)
        input2 = input2.unsqueeze(1)   # Shape: (batch_size, 1, dim2)
        
        if self.internal_attn:
            input1 = input1.transpose(1,2) # Shape: (batch_size, dim1, 1)
            input2 = input2.transpose(1,2) # Shape: (batch_size, dim2, 1)

        # Cross-attention: input1 attends to input2
        output1, attn_weights_1 = self.x_attn(input1, input2, input2)  # Query: input1, Key/Value: input2
        
        # Cross-attention: input2 attends to input1
        output2, attn_weights_2 = self.x_attn(input2, input1, input1)  # Query: input2, Key/Value: input1

        if self.internal_attn:
            output1 = output1.transpose(1,2)
            output2 = output2.transpose(1,2)

        # Remove sequence dimension
        output1 = self.dropout(output1).squeeze(1)  # Shape: (batch_size, hidden_dim)
        output2 = self.dropout(output2).squeeze(1)  # Shape: (batch_size, hidden_dim)

        if return_attn:
            return (output1, output2), (attn_weights_1, attn_weights_2)
        else:
            return output1, output2
    
    
class FusionModel(nn.Module):
    def __init__(self, 
                text_encoder_id: str = "dmis-lab/biobert-v1.1", 
                gene_input_dim: int = 512,
                hidden_size: int = 256,
                attention_heads: int = 8,
                dropout: float = 0.1,
                freeze_text_encoder: bool = True,
                internal_attn: bool = False,
                add_cross_attn: bool = True,
        ):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=text_encoder_id)
        self.add_cross_attn = add_cross_attn
        self.hidden_size = hidden_size
        if add_cross_attn:
            self.x_attn = CrossModalAttention(
                hidden_dim=hidden_size, 
                dim1=self.text_encoder.text_hidden_dim, 
                dim2=gene_input_dim, 
                num_heads=attention_heads,
                internal_attn=internal_attn,
                dropout=dropout,
            )
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
    
    def forward(self, x):
        text_input, gene_emb = x
        text_emb = self.text_encoder(text_input)  # (batch_size, text_hidden_dim)
        if self.add_cross_attn:
            text_emb, gene_emb = self.x_attn(text_emb, gene_emb)  # (batch_size, hidden_size)
        return text_emb, gene_emb           # two modals in the common latent space


class UnifiedMultiModalClassifier(FusionModel):
    """
    Main Model for multimodal donor prediction.
    
    Process flow:
      1. Process gene (omics) data through PerceiverResampler to obtain a sequence
         of latent tokens.
      2. Process text data via TextEncoderModule to obtain token-level embeddings,
         then project them to the common latent dimension.
      3. Use CrossModalAttention to fuse the gene and text modalities.
      4. Use an MLP classifier on the fused representation for donor prediction.
    """
    def __init__(self,
                class_num: int,
                dropout: float = 0.3,
                *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + 1, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),          # regularization to prevent overfitting
            nn.Linear(self.hidden_size, class_num)
        )
        
        # self.adversary = AdversarialModel(self.hidden_size)  # Adversary predicts sex or other confounders
        # self.lambda_adv = lambda_adv
        

    def forward(self, x):
        text_input, gene_emb, sex = x
        text_emb, gene_emb = super().forward((text_input, gene_emb))  # apply cross-modal fusion
        final_vec = torch.cat([text_emb, gene_emb, sex.unsqueeze(1)], dim=1)  # Concatenate text and gene embeddings
        logits = self.classification_head(final_vec)    # Classification head
        
        # # Adversarial prediction (detach to avoid influencing main model)
        # adv_input = final_vec[:, :-1]  # Remove `sex` before passing to adversary
        # adversary_logits = self.adversary(GradientReversalLayer.apply(adv_input, reverse_lambda))
        
        # return logits, adversary_logits
        
        return logits


    def from_pretrained(self, path: str):
        state_dict = torch.load(path, map_location='cpu')
        tgt_params = self.state_dict()
        for param, value in state_dict.items():
            tgt_params[param] = value
        self.load_state_dict(tgt_params)
        
        
class AdversarialModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)  # Binary classification for sex confounders
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # predict logits
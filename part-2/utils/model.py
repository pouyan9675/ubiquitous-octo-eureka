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


class PerceiverResampler(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_latent_vectors=64, 
                 latent_dim=256, 
                 num_heads=8):
        super().__init__()
        
        # Trainable latent vectors
        self.latent_vectors = nn.Parameter(torch.randn(num_latent_vectors, latent_dim))
        
        # Projection layer to align input dimension
        self.input_projection = nn.Linear(input_dim, latent_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim, 
            num_heads=num_heads
        )
        
        # Layer normalization and feed-forward
        self.layer_norm1 = nn.LayerNorm(latent_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
    
    def forward(self, input_embeddings):
        # Project input to latent dimension
        input_embeddings = self.input_projection(input_embeddings)
        
        # Prepare latent vectors and input
        batch_size = input_embeddings.size(0)
        latent_vectors = self.latent_vectors.unsqueeze(1).repeat(1, batch_size, 1)
        input_embeddings = input_embeddings.unsqueeze(0)
        
        # Cross-attention
        attn_output, attn_weights = self.cross_attention(
            latent_vectors,  # query
            input_embeddings,  # key
            input_embeddings  # value
        )
        
        # Residual connection and layer norm
        attn_output = self.layer_norm1(attn_output + latent_vectors)
        
        # Feed-forward
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm2(ff_output + attn_output)
        
        return output.transpose(0, 1), attn_weights


class CrossModalAttention(nn.Module):
    """
    A module that applied perceiver resample to both modalities and then performs a bi-directional
    cross-attention to each modal attend to the other modality.

    Bi-directional Cross Attention:
    Given two input feature tensors with different dimensions (dim1 and dim2), this module applies:
    - Perceiver Resampler applied to input1 -> (batch_size, num_latent_vectors, hidden_dim)
    - Perceiver Resampler applied to input2 -> (batch_size, num_latent_vectors, hidden_dim)
    - Attention from input1 (query) attending to input2 (key, value).
    - Attention from input2 (query) attending to input1 (key, value).
    - Apply mean pooling to both sequences -> (batch_size, hidden_dim)
    
    Parameters:
    -----------
    dim1 : int
        Embedding dimension of input1.
    dim2 : int
        Embedding dimension of input2.
    num_heads : int
        Number of attention heads.

    Returns:
    --------
    output1 : torch.Tensor
        Cross-attended representation of input1 (batch_size, hidden_size).
    output2 : torch.Tensor
        Cross-attended representation of input2 (batch_size, hidden_size).
    """

    def __init__(self, 
                hidden_dim: int,
                dim1: int,
                dim2: int,
                num_heads: int = 8,
                num_latent_vectors: int = 64,
                dropout: float = 0.1,
    ):
        super().__init__()
        self.text_perceiver = PerceiverResampler(
            input_dim=dim1,
            num_latent_vectors=num_latent_vectors,
            latent_dim=hidden_dim,
            num_heads=num_heads
        )
        
        self.gene_perceiver = PerceiverResampler(
            input_dim=dim2,
            num_latent_vectors=num_latent_vectors,
            latent_dim=hidden_dim,
            num_heads=num_heads
        )
        
        self.x_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Add normalization layers for each modality
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, input1: torch.Tensor, input2: torch.Tensor, return_attn: bool = False):
        # Apply Perceiver Resamplers to inputs
        text_latents, text_attn = self.text_perceiver(input1)  # [batch, num_latent_vectors, hidden_dim]
        gene_latents, gene_attn = self.gene_perceiver(input2)  # [batch, num_latent_vectors, hidden_dim]
        
        # Apply dropout
        text_latents = self.dropout(text_latents)
        gene_latents = self.dropout(gene_latents)
        
        # I have applied cross attention bi-directional as gene and cell_type
        # do not have any priorty over each other to apply it just in one way
        
        # Cross-attention: text attends to gene
        # Query: text_latents, Key/Value: gene_latents
        text_attended, cross_attn_weights_1 = self.x_attn(
            text_latents, gene_latents, gene_latents
        )
        # Apply residual connection and layer norm
        text_latents = self.norm1(text_latents + self.dropout(text_attended))
        
        # Cross-attention: gene attends to text
        # Query: gene_latents, Key/Value: text_latents
        gene_attended, cross_attn_weights_2 = self.x_attn(
            gene_latents, text_latents, text_latents
        )
        # Apply residual connection and layer norm
        gene_latents = self.norm2(gene_latents + self.dropout(gene_attended))
        
        # Apply pooling to get a single vector per modality
        # Here a self-attention with a single trainable query can be used to 
        # have a smarter pooling but due to simpler implementation I just use
        # a single mean pooling.
        text_pooled = torch.mean(text_attended, dim=1)  # [batch, hidden_dim]
        gene_pooled = torch.mean(gene_attended, dim=1)  # [batch, hidden_dim]
        
        if return_attn:
            return (text_pooled, gene_pooled), (cross_attn_weights_1, cross_attn_weights_2)
        else:
            return text_pooled, gene_pooled
    
    
class FusionModel(nn.Module):
    def __init__(self, 
                text_encoder_id: str = "dmis-lab/biobert-v1.1", 
                gene_input_dim: int = 512,
                hidden_size: int = 256,
                attention_heads: int = 8,
                num_latent_vectors: int = 64,
                dropout: float = 0.1,
                freeze_text_encoder: bool = True,
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
                num_latent_vectors=num_latent_vectors,
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
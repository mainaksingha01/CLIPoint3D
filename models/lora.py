import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class LoRALinear(nn.Module):
    """
    LoRA (Low Rank Adaptation) wrapper for linear layers.
    
    Args:
        original_linear: The original linear layer to wrap
        rank: Rank of the low-rank adaptation (default: 16)
        alpha: LoRA scaling parameter (default: 32)
        dropout: Dropout rate for LoRA layers (default: 0.1)
    """
    def __init__(self, 
                 original_linear: nn.Linear, 
                 rank: int = 16, 
                 alpha: int = 32, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False
            
        # Create LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, original_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_linear.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Original computation
        result = self.original_linear(x)
        
        # LoRA adaptation: x @ (A^T @ B^T) * scaling
        # Cast LoRA parameters to input dtype
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        lora_result = (self.dropout(x) @ lora_A.T @ lora_B.T) * self.scaling
        
        return result + lora_result
        
    def merge_weights(self):
        """Merge LoRA weights into the original linear layer for inference."""
        if hasattr(self, '_merged') and self._merged:
            return
            
        # Merge: W_new = W_original + alpha/rank * B @ A
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.original_linear.weight.data += delta_weight
        self._merged = True
        
    def unmerge_weights(self):
        """Unmerge LoRA weights from the original linear layer."""
        if not (hasattr(self, '_merged') and self._merged):
            return
            
        # Unmerge: W_original = W_new - alpha/rank * B @ A
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.original_linear.weight.data -= delta_weight
        self._merged = False


class LoRAMultiheadAttention(nn.Module):
    """
    Simplified LoRA wrapper for MultiheadAttention layer.
    Applies LoRA only to the output projection for simplicity.
    """
    def __init__(self, 
                 original_attention: nn.MultiheadAttention,
                 rank: int = 16,
                 alpha: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        
        self.original_attention = original_attention
        self.embed_dim = original_attention.embed_dim
        self.num_heads = original_attention.num_heads
        
        # Freeze original attention weights
        for param in self.original_attention.parameters():
            param.requires_grad = False
            
        # LoRA for output projection only (simplified approach)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices for output projection
        self.lora_out_proj_A = nn.Parameter(torch.randn(rank, self.embed_dim) * 0.01)
        self.lora_out_proj_B = nn.Parameter(torch.zeros(self.embed_dim, rank))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        # Use original attention mechanism
        attn_output, attn_output_weights = self.original_attention(
            query, key, value, 
            need_weights=need_weights, 
            attn_mask=attn_mask
        )
        
        # Apply LoRA to output (cast to input dtype)
        lora_out_proj_A = self.lora_out_proj_A.to(attn_output.dtype)
        lora_out_proj_B = self.lora_out_proj_B.to(attn_output.dtype)
        lora_out = (self.dropout(attn_output) @ lora_out_proj_A.T @ lora_out_proj_B.T) * self.scaling
        attn_output = attn_output + lora_out
        
        if need_weights:
            return attn_output, attn_output_weights
        return attn_output, None
        
    def merge_weights(self):
        """Merge LoRA weights into the original attention layer for inference."""
        if hasattr(self, '_merged') and self._merged:
            return
            
        # Merge LoRA weights into output projection
        delta_weight = (self.lora_out_proj_B @ self.lora_out_proj_A) * self.scaling
        self.original_attention.out_proj.weight.data += delta_weight
        self._merged = True
        
    def unmerge_weights(self):
        """Unmerge LoRA weights from the original attention layer."""
        if not (hasattr(self, '_merged') and self._merged):
            return
            
        # Unmerge LoRA weights from output projection  
        delta_weight = (self.lora_out_proj_B @ self.lora_out_proj_A) * self.scaling
        self.original_attention.out_proj.weight.data -= delta_weight
        self._merged = False


class LoRAImageEncoder(nn.Module):
    """
    LoRA wrapper for ImageEncoder that applies LoRA to transformer attention and MLP layers.
    
    Args:
        image_encoder: Original ImageEncoder instance
        rank: Rank for LoRA adaptation (default: 16)
        alpha: LoRA scaling parameter (default: 32)
        dropout: Dropout rate for LoRA layers (default: 0.1)
        target_modules: List of module types to apply LoRA to (default: ['attn', 'mlp'])
    """
    def __init__(self, 
                 image_encoder,
                 rank: int = 16,
                 alpha: int = 32,
                 dropout: float = 0.1,
                 target_modules: List[str] = ['attn', 'mlp']):
        super().__init__()
        
        self.original_encoder = image_encoder
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        
        # Copy all non-transformer attributes
        self.conv1 = image_encoder.conv1
        self.class_embedding = image_encoder.class_embedding
        self.positional_embedding = image_encoder.positional_embedding
        self.ln_pre = image_encoder.ln_pre
        self.ln_post = image_encoder.ln_post
        self.proj = image_encoder.proj
        
        # Freeze non-transformer parameters
        for param in [self.conv1, self.class_embedding, self.positional_embedding, 
                      self.ln_pre, self.ln_post, self.proj]:
            if hasattr(param, 'parameters'):
                for p in param.parameters():
                    p.requires_grad = False
            elif isinstance(param, nn.Parameter):
                param.requires_grad = False
        
        # Apply LoRA to transformer blocks
        self.transformer = self._apply_lora_to_transformer(image_encoder.transformer)
        
    def _apply_lora_to_transformer(self, transformer):
        """Apply LoRA to transformer blocks."""
        # Create new transformer with LoRA-wrapped blocks
        modified_blocks = []
        
        for block in transformer.resblocks:
            modified_block = self._wrap_residual_attention_block(block)
            modified_blocks.append(modified_block)
            
        # Create new transformer with modified blocks
        new_transformer = nn.Module()
        new_transformer.resblocks = nn.ModuleList(modified_blocks)
        new_transformer.width = transformer.width
        new_transformer.layers = transformer.layers
        
        # Add forward method to the transformer
        def transformer_forward(self, x):
            for block in self.resblocks:
                x = block(x)
            return x
        
        new_transformer.forward = transformer_forward.__get__(new_transformer)
        
        return new_transformer
    
    def _wrap_residual_attention_block(self, block):
        """Wrap a ResidualAttentionBlock with LoRA."""
        # Create a new block that wraps the original
        wrapped_block = nn.Module()
        
        # Copy layer norms (frozen)
        wrapped_block.ln_1 = block.ln_1
        wrapped_block.ln_2 = block.ln_2
        wrapped_block.attn_mask = block.attn_mask
        
        # Freeze layer norm parameters
        for param in wrapped_block.ln_1.parameters():
            param.requires_grad = False
        for param in wrapped_block.ln_2.parameters():
            param.requires_grad = False
        
        # Wrap attention with LoRA if requested
        if 'attn' in self.target_modules:
            wrapped_block.attn = LoRAMultiheadAttention(
                block.attn, self.rank, self.alpha, self.dropout
            )
        else:
            wrapped_block.attn = block.attn
            
        # Wrap MLP layers with LoRA if requested
        if 'mlp' in self.target_modules:
            wrapped_block.mlp = nn.Sequential()
            for name, layer in block.mlp.named_children():
                if isinstance(layer, nn.Linear):
                    wrapped_block.mlp.add_module(name, LoRALinear(
                        layer, self.rank, self.alpha, self.dropout
                    ))
                else:
                    wrapped_block.mlp.add_module(name, layer)
        else:
            wrapped_block.mlp = block.mlp
            
        # Add the attention method
        def attention(self, x):
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        
        wrapped_block.attention = attention.__get__(wrapped_block)
        
        # Add the forward method
        def forward(self, x):
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
            
        wrapped_block.forward = forward.__get__(wrapped_block)
        
        return wrapped_block
    
    def insert_prompts(self, x, prompt_embeddings):
        """Insert prompts (from original encoder)."""
        return self.original_encoder.insert_prompts(x, prompt_embeddings)
    
    def pre(self, x):
        """Pre-processing (from original encoder)."""
        return self.original_encoder.pre(x)
    
    def post(self, x):
        """Post-processing (from original encoder)."""
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    
    def forward(self, x):
        """Forward pass with LoRA adaptations."""
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # insert prompts here if needed
       
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for optimization."""
        lora_params = []
        for module in self.modules():
            if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
                for name, param in module.named_parameters():
                    if 'lora_' in name:
                        lora_params.append(param)
        return lora_params
    
    def merge_and_unload(self):
        """Merge LoRA weights and return the original encoder with merged weights."""
        # Merge all LoRA weights
        for module in self.modules():
            if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
                module.merge_weights()
        
        # Return the original encoder (now with merged weights)
        return self.original_encoder
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights to file."""
        lora_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
                for param_name, param in module.named_parameters():
                    if 'lora_' in param_name:
                        lora_state_dict[f"{name}.{param_name}"] = param.data
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'rank': self.rank,
            'alpha': self.alpha,
            'target_modules': self.target_modules
        }, path)
        
    def load_lora_weights(self, path: str):
        """Load LoRA weights from file."""
        checkpoint = torch.load(path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        
        # Load the weights
        for name, param in lora_state_dict.items():
            module_path, param_name = name.rsplit('.', 1)
            module = self
            for attr in module_path.split('.'):
                module = getattr(module, attr)
            getattr(module, param_name).data.copy_(param)


class LoRATextEncoder(nn.Module):
    """
    LoRA wrapper for TextEncoder that applies LoRA to transformer attention and MLP layers.
    
    Args:
        text_encoder: Original TextEncoder instance
        rank: Rank for LoRA adaptation (default: 16)
        alpha: LoRA scaling parameter (default: 32)
        dropout: Dropout rate for LoRA layers (default: 0.1)
        target_modules: List of module types to apply LoRA to (default: ['attn', 'mlp'])
    """
    def __init__(self, 
                 text_encoder,
                 rank: int = 16,
                 alpha: int = 32,
                 dropout: float = 0.1,
                 target_modules: List[str] = ['attn', 'mlp']):
        super().__init__()
        
        self.original_encoder = text_encoder
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        
        # Copy all non-transformer attributes
        self.positional_embedding = text_encoder.positional_embedding
        self.ln_final = text_encoder.ln_final
        self.text_projection = text_encoder.text_projection
        self.dtype = text_encoder.dtype
        
        # Freeze non-transformer parameters
        for param in [self.positional_embedding, self.ln_final, self.text_projection]:
            if hasattr(param, 'parameters'):
                for p in param.parameters():
                    p.requires_grad = False
            elif isinstance(param, nn.Parameter):
                param.requires_grad = False
        
        # Apply LoRA to transformer blocks
        self.transformer = self._apply_lora_to_transformer(text_encoder.transformer)
        
    def _apply_lora_to_transformer(self, transformer):
        """Apply LoRA to transformer blocks."""
        # Create new transformer with LoRA-wrapped blocks
        modified_blocks = []
        
        for block in transformer.resblocks:
            modified_block = self._wrap_residual_attention_block(block)
            modified_blocks.append(modified_block)
            
        # Create new transformer with modified blocks
        new_transformer = nn.Module()
        new_transformer.resblocks = nn.ModuleList(modified_blocks)
        new_transformer.width = transformer.width
        new_transformer.layers = transformer.layers
        
        # Add forward method to the transformer
        def transformer_forward(self, x):
            for block in self.resblocks:
                x = block(x)
            return x
        
        new_transformer.forward = transformer_forward.__get__(new_transformer)
        
        return new_transformer
    
    def _wrap_residual_attention_block(self, block):
        """Wrap a ResidualAttentionBlock with LoRA."""
        # Create a new block that wraps the original
        wrapped_block = nn.Module()
        
        # Copy layer norms (frozen)
        wrapped_block.ln_1 = block.ln_1
        wrapped_block.ln_2 = block.ln_2
        wrapped_block.attn_mask = block.attn_mask
        
        # Freeze layer norm parameters
        for param in wrapped_block.ln_1.parameters():
            param.requires_grad = False
        for param in wrapped_block.ln_2.parameters():
            param.requires_grad = False
        
        # Wrap attention with LoRA if requested
        if 'attn' in self.target_modules:
            wrapped_block.attn = LoRAMultiheadAttention(
                block.attn, self.rank, self.alpha, self.dropout
            )
        else:
            wrapped_block.attn = block.attn
            
        # Wrap MLP layers with LoRA if requested
        if 'mlp' in self.target_modules:
            wrapped_block.mlp = nn.Sequential()
            for name, layer in block.mlp.named_children():
                if isinstance(layer, nn.Linear):
                    wrapped_block.mlp.add_module(name, LoRALinear(
                        layer, self.rank, self.alpha, self.dropout
                    ))
                else:
                    wrapped_block.mlp.add_module(name, layer)
        else:
            wrapped_block.mlp = block.mlp
            
        # Add the attention method
        def attention(self, x):
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        
        wrapped_block.attention = attention.__get__(wrapped_block)
        
        # Add the forward method
        def forward(self, x):
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
            
        wrapped_block.forward = forward.__get__(wrapped_block)
        
        return wrapped_block
    
    def forward(self, prompts, tokenized_prompts):
        """Forward pass with LoRA adaptations."""
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].type(self.dtype) @ self.text_projection.type(self.dtype)

        return x
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for optimization."""
        lora_params = []
        for module in self.modules():
            if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
                for name, param in module.named_parameters():
                    if 'lora_' in name:
                        lora_params.append(param)
        return lora_params
    
    def merge_and_unload(self):
        """Merge LoRA weights and return the original encoder with merged weights."""
        # Merge all LoRA weights
        for module in self.modules():
            if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
                module.merge_weights()
        
        # Return the original encoder (now with merged weights)
        return self.original_encoder
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights to file."""
        lora_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
                for param_name, param in module.named_parameters():
                    if 'lora_' in param_name:
                        lora_state_dict[f"{name}.{param_name}"] = param.data
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'rank': self.rank,
            'alpha': self.alpha,
            'target_modules': self.target_modules
        }, path)
        
    def load_lora_weights(self, path: str):
        """Load LoRA weights from file."""
        checkpoint = torch.load(path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        
        # Load the weights
        for name, param in lora_state_dict.items():
            module_path, param_name = name.rsplit('.', 1)
            module = self
            for attr in module_path.split('.'):
                module = getattr(module, attr)
            getattr(module, param_name).data.copy_(param)


def create_lora_image_encoder(original_encoder, 
                             rank: int = 16,
                             alpha: int = 32,
                             dropout: float = 0.1,
                             target_modules: List[str] = ['attn', 'mlp']):
    """
    Convenience function to create a LoRA-wrapped ImageEncoder.
    
    Args:
        original_encoder: The original ImageEncoder instance
        rank: Rank for LoRA adaptation (default: 16)
        alpha: LoRA scaling parameter (default: 32)  
        dropout: Dropout rate for LoRA layers (default: 0.1)
        target_modules: List of module types to apply LoRA to (default: ['attn', 'mlp'])
        
    Returns:
        LoRAImageEncoder instance
    """
    return LoRAImageEncoder(
        original_encoder, 
        rank=rank, 
        alpha=alpha, 
        dropout=dropout, 
        target_modules=target_modules
    )


def create_lora_text_encoder(original_encoder, 
                            rank: int = 16,
                            alpha: int = 32,
                            dropout: float = 0.1,
                            target_modules: List[str] = ['attn', 'mlp']):
    """
    Convenience function to create a LoRA-wrapped TextEncoder.
    
    Args:
        original_encoder: The original TextEncoder instance
        rank: Rank for LoRA adaptation (default: 16)
        alpha: LoRA scaling parameter (default: 32)  
        dropout: Dropout rate for LoRA layers (default: 0.1)
        target_modules: List of module types to apply LoRA to (default: ['attn', 'mlp'])
        
    Returns:
        LoRATextEncoder instance
    """
    return LoRATextEncoder(
        original_encoder, 
        rank=rank, 
        alpha=alpha, 
        dropout=dropout, 
        target_modules=target_modules
    ) 
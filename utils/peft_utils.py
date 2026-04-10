"""
Parameter Efficient Fine-Tuning (PEFT) utilities for the custom Model architecture.
This module provides functions for BitFit and LayerNorm-only fine-tuning.
Adapted to work with the specific model structure using LoRA ImageEncoder.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Any


def named_modules_with_index(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module, int]]:
    """
    Walk through model modules and assign block indices for PEFT methods.
    Adapted for the custom Model class structure.
    
    Args:
        model: Custom Model class or its components (image_encoder, text_encoder)
        
    Returns:
        List of tuples (name, module, block_idx) where block_idx represents
        the transformer block index for vision/text encoders
    """
    modules_with_index = []
    
    for name, module in model.named_modules():
        # Determine block index based on module name patterns
        block_idx = 0
        
        # For LoRA image encoder transformer blocks
        if 'transformer.resblocks' in name and any(x in name for x in ['image_encoder', 'visual']):
            # Extract block number from name like "image_encoder.transformer.resblocks.0.ln_1"
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'resblocks' and i + 1 < len(parts):
                    try:
                        block_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass
        
        # For text encoder transformer blocks
        elif 'text_encoder.transformer.resblocks' in name:
            # Extract block number from name like "text_encoder.transformer.resblocks.0.ln_1"
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'resblocks' and i + 1 < len(parts):
                    try:
                        block_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass
        
        # For ResNet-based image encoder layers (if using ResNet backbone)
        elif any(x in name for x in ['image_encoder.layer', 'visual.layer']):
            # Extract layer number from name like "image_encoder.layer1.0.conv1"
            parts = name.split('.')
            for part in parts:
                if part.startswith('layer'):
                    try:
                        block_idx = int(part.replace('layer', '')) - 1  # 0-indexed
                        break
                    except ValueError:
                        pass
        
        # For attention pooling in ResNet image encoder
        elif any(x in name for x in ['image_encoder.attnpool', 'visual.attnpool']):
            block_idx = 4  # After all ResNet layers
        
        # For LoRA layers in image encoder
        elif 'image_encoder' in name and 'lora_' in name:
            # Try to extract transformer block index for LoRA layers
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part.isdigit():
                    try:
                        block_idx = int(part)
                        break
                    except ValueError:
                        pass
        
        modules_with_index.append((name, module, block_idx))
    
    return modules_with_index


def trainable_norm_params(model: torch.nn.Module, vision_start: int = 0) -> List[torch.nn.Parameter]:
    """
    Make only LayerNorm parameters trainable (ln_only fine-tuning method).
    Works with image encoder modules only.
    
    Args:
        model: ImageEncoder module
        vision_start: Starting block index for vision encoder
        
    Returns:
        List of trainable parameters
    """
    trainable_params = []
    
    for name, module, block_idx in named_modules_with_index(model):
        # Check if this is a LayerNorm module and meets the criteria
        if (isinstance(module, (torch.nn.LayerNorm, nn.LayerNorm)) and 
            block_idx >= vision_start):
            
            # Make LayerNorm parameters trainable
            for param in module.parameters():
                param.requires_grad_(True)
                trainable_params.append(param)
            print(f"LayerNorm at {name} is trainable.")
    
    return trainable_params


def trainable_bias_params(model: torch.nn.Module, vision_start: int = 0) -> List[torch.nn.Parameter]:
    """
    Make only bias parameters trainable (BitFit fine-tuning method).
    Works with image encoder modules only.
    
    Args:
        model: ImageEncoder module
        vision_start: Starting block index for vision encoder
        
    Returns:
        List of trainable parameters
    """
    trainable_params = []

    for name, module, block_idx in named_modules_with_index(model):
        # Check if this module has bias and meets the criteria
        if (hasattr(module, "bias") and module.bias is not None and
            block_idx >= vision_start):
            
            # Make bias parameter trainable
            module.bias.requires_grad_(True)
            trainable_params.append(module.bias)
            print(f"Bias at {name}.bias is trainable.")
    return trainable_params


def get_peft_params_count(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Get count of trainable and total parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params


def print_peft_summary(model: torch.nn.Module):
    """
    Print summary of parameter-efficient fine-tuning setup.
    
    Args:
        model: PyTorch model
    """
    trainable, total = get_peft_params_count(model)
    percentage = (trainable / total) * 100 if total > 0 else 0
    
    print(f"\n=== PEFT Summary ===")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Percentage of trainable parameters: {percentage:.4f}%")
    print("=" * 20)


# Wrapper classes for PEFT methods (similar to LoRA pattern)
class BitFitImageEncoder(nn.Module):
    """
    BitFit wrapper for ImageEncoder that applies bias-only fine-tuning.
    Follows the same pattern as LoRAImageEncoder for integration with trainer.
    """
    def __init__(self, image_encoder, vision_start: int = 0):
        super().__init__()
        self.original_encoder = image_encoder
        self.vision_start = vision_start
        
        # Apply BitFit to the encoder
        self._apply_bitfit()
    
    def _apply_bitfit(self):
        """Apply BitFit fine-tuning to the encoder."""
        # First freeze all parameters
        for param in self.original_encoder.parameters():
            param.requires_grad_(False)
        
        # Then enable bias parameters based on criteria
        self.bitfit_params = trainable_bias_params(
            self.original_encoder, 
            self.vision_start
        )
        print(f"BitFit: Found {len(self.bitfit_params)} bias parameters in image encoder")
    
    def get_bitfit_parameters(self):
        """Get all BitFit parameters for optimization (follows LoRA pattern)."""
        return self.bitfit_params
    
    def forward(self, *args, **kwargs):
        """Forward pass through original encoder."""
        return self.original_encoder(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to original encoder."""
        if name in ['original_encoder', 'vision_start', 'bitfit_params']:
            return super().__getattr__(name)
        try:
            return getattr(self.original_encoder, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class LayerNormOnlyImageEncoder(nn.Module):
    """
    LayerNorm-only wrapper for ImageEncoder that applies LN-only fine-tuning.
    Follows the same pattern as LoRAImageEncoder for integration with trainer.
    """
    def __init__(self, image_encoder, vision_start: int = 0):
        super().__init__()
        self.original_encoder = image_encoder
        self.vision_start = vision_start
        
        # Apply LayerNorm-only fine-tuning
        self._apply_ln_only()
    
    def _apply_ln_only(self):
        """Apply LayerNorm-only fine-tuning to the encoder."""
        # First freeze all parameters
        for param in self.original_encoder.parameters():
            param.requires_grad_(False)
        
        # Then enable LayerNorm parameters based on criteria
        self.ln_params = trainable_norm_params(
            self.original_encoder,
            self.vision_start
        )
        print(f"LayerNorm-only: Found {len(self.ln_params)} LayerNorm parameters in image encoder")
    
    def get_ln_only_parameters(self):
        """Get all LayerNorm parameters for optimization (follows LoRA pattern)."""
        return self.ln_params
    
    def forward(self, *args, **kwargs):
        """Forward pass through original encoder."""
        return self.original_encoder(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to original encoder."""
        if name in ['original_encoder', 'vision_start', 'ln_params']:
            return super().__getattr__(name)
        try:
            return getattr(self.original_encoder, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class BitFitTextEncoder(nn.Module):
    """
    BitFit wrapper for TextEncoder that applies bias-only fine-tuning.
    Follows the same pattern as BitFitImageEncoder for integration with trainer.
    """
    def __init__(self, text_encoder, text_start: int = 0):
        super().__init__()
        self.original_encoder = text_encoder
        self.text_start = text_start
        
        # Apply BitFit to the encoder
        self._apply_bitfit()
    
    def _apply_bitfit(self):
        """Apply BitFit fine-tuning to the encoder."""
        # First freeze all parameters
        for param in self.original_encoder.parameters():
            param.requires_grad_(False)
        
        # Then enable bias parameters based on criteria
        self.bitfit_params = trainable_bias_params(
            self.original_encoder, 
            self.text_start
        )
        print(f"BitFit: Found {len(self.bitfit_params)} bias parameters in text encoder")
    
    def get_bitfit_parameters(self):
        """Get all BitFit parameters for optimization (follows LoRA pattern)."""
        return self.bitfit_params
    
    def forward(self, *args, **kwargs):
        """Forward pass through original encoder."""
        return self.original_encoder(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to original encoder."""
        if name in ['original_encoder', 'text_start', 'bitfit_params']:
            return super().__getattr__(name)
        try:
            return getattr(self.original_encoder, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class LayerNormOnlyTextEncoder(nn.Module):
    """
    LayerNorm-only wrapper for TextEncoder that applies LN-only fine-tuning.
    Follows the same pattern as LayerNormOnlyImageEncoder for integration with trainer.
    """
    def __init__(self, text_encoder, text_start: int = 0):
        super().__init__()
        self.original_encoder = text_encoder
        self.text_start = text_start
        
        # Apply LayerNorm-only fine-tuning
        self._apply_ln_only()
    
    def _apply_ln_only(self):
        """Apply LayerNorm-only fine-tuning to the encoder."""
        # First freeze all parameters
        for param in self.original_encoder.parameters():
            param.requires_grad_(False)
        
        # Then enable LayerNorm parameters based on criteria
        self.ln_params = trainable_norm_params(
            self.original_encoder,
            self.text_start
        )
        print(f"LayerNorm-only: Found {len(self.ln_params)} LayerNorm parameters in text encoder")
    
    def get_ln_only_parameters(self):
        """Get all LayerNorm parameters for optimization (follows LoRA pattern)."""
        return self.ln_params
    
    def forward(self, *args, **kwargs):
        """Forward pass through original encoder."""
        return self.original_encoder(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to original encoder."""
        if name in ['original_encoder', 'text_start', 'ln_params']:
            return super().__getattr__(name)
        try:
            return getattr(self.original_encoder, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Factory functions for creating PEFT encoders
def create_bitfit_image_encoder(original_encoder, **kwargs):
    """Create a BitFit-wrapped ImageEncoder."""
    return BitFitImageEncoder(original_encoder, **kwargs)


def create_ln_only_image_encoder(original_encoder, **kwargs):
    """Create a LayerNorm-only wrapped ImageEncoder."""
    return LayerNormOnlyImageEncoder(original_encoder, **kwargs)


def create_bitfit_text_encoder(original_encoder, **kwargs):
    """Create a BitFit-wrapped TextEncoder."""
    return BitFitTextEncoder(original_encoder, **kwargs)


def create_ln_only_text_encoder(original_encoder, **kwargs):
    """Create a LayerNorm-only wrapped TextEncoder."""
    return LayerNormOnlyTextEncoder(original_encoder, **kwargs)


# Convenience functions for common PEFT setups
def setup_bitfit(model: torch.nn.Module, **kwargs) -> List[torch.nn.Parameter]:
    """Setup BitFit (bias-only fine-tuning)."""
    params = trainable_bias_params(model, **kwargs)
    print_peft_summary(model)
    return params


def setup_ln_only(model: torch.nn.Module, **kwargs) -> List[torch.nn.Parameter]:
    """Setup LayerNorm-only fine-tuning."""
    params = trainable_norm_params(model, **kwargs)
    print_peft_summary(model)
    return params

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .constants import CLASS_MAP, CLASS_POLYMER_COATING, CLASS_VOIDS


class PhysicsInformedCompositeLoss(nn.Module):
    """
    A loss function that combines a baseline loss with a physics-informed penalty for Voids and Polymer Coating classes. 
    
    The physics-informed penalty is designed to encourage more realistic predictions by penalizing the predicted masks based on their morphological properties.
    
    Uses the CLASS_MAP to determine class indices.
    """
    def __init__(
        self, 
        baseline: Optional[nn.Module] = None,
        baseline_weight: float = 1.0, 
        physics_weight: float = 0.1
    ):
        """
        Initialize the PhysicsInformedCompositeLoss.
        
        It depends on the CLASS_MAP (`constants.py`) to determine the indices of the void and polymer coating classes, 
        and applies a physics-informed penalty based on the predicted masks for these classes.
        
        Args:
            baseline (nn.Module | None): The baseline loss function, if None, defaults to CrossEntropyLoss.
            baseline_weight (float): Weight for the baseline loss.
            physics_weight (float): Weight for the physics-informed penalty.
        """
        super().__init__()
        self.physics_weight = physics_weight
        self.baseline_weight = baseline_weight
        
        self.void_class_index = CLASS_MAP[CLASS_VOIDS]
        self.polymer_coating_class_index = CLASS_MAP[CLASS_POLYMER_COATING]
        
        self.base_loss = baseline if baseline is not None else nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        base_loss = self.base_loss(logits, targets)
        
        probs = F.softmax(logits, dim=1)
        physics_penalty = 0.0
        
        class_probs_voids = probs[:, self.void_class_index, :, :]
        class_probs_polymer = probs[:, self.polymer_coating_class_index, :, :]
        
        # Calculate spatial gradients to detect predicted perimeters per class
        dy_voids = torch.abs(class_probs_voids[:, 1:, :] - class_probs_voids[:, :-1, :])
        dx_voids = torch.abs(class_probs_voids[:, :, 1:] - class_probs_voids[:, :, :-1])
        
        dy_polymer = torch.abs(class_probs_polymer[:, 1:, :] - class_probs_polymer[:, :-1, :])
        dx_polymer = torch.abs(class_probs_polymer[:, :, 1:] - class_probs_polymer[:, :, :-1])
        
        perimeter_voids = torch.sum(dx_voids) + torch.sum(dy_voids)
        perimeter_polymer = torch.sum(dx_polymer) + torch.sum(dy_polymer)
        
        # penalty for voids: scale-invariant compactness
        area_voids = torch.sum(class_probs_voids) + 1e-6
        # perimeter_sq_voids = perimeter_voids ** 2
        physics_penalty += (perimeter_voids / area_voids)
        
        # penalty for polymer coating: pure total variation
        num_pixels_polymer = class_probs_polymer.shape[1] * class_probs_polymer.shape[2]
        physics_penalty += (perimeter_polymer / num_pixels_polymer)
        
        total_loss = (self.baseline_weight * base_loss) + (self.physics_weight * physics_penalty)
        
        return total_loss

    def to_dict(self):
        return {
            "baseline": self.base_loss.__class__.__name__,
            "baseline_weight": self.baseline_weight,
            "physics_weight": self.physics_weight,
        }


class PhysicsInformedVoidsLoss(nn.Module):
    """
    Loss that combines a baseline loss function with a physics-informed penalty based specifically on the predicted void class.
    """
    def __init__(
        self, 
        baseline: Optional[nn.Module] = None,
        baseline_weight: float = 1.0,
        physics_weight: float = 0.1, 
    ):
        """
        Initialize the PhysicsInformedVoidsLoss.
        
        Uses the CLASS_MAP (`constants.py`) to identify the index of the void class and applies a physics-informed penalty based on the predicted voids.

        Args:
            baseline (nn.Module): The baseline loss function. If None, defaults to CrossEntropyLoss.
            baseline_weight (float): The weight for the baseline loss.
            physics_weight (float): The weight for the physics-informed penalty.
        """
        super().__init__()
        self.physics_weight = physics_weight
        self.void_class_index = CLASS_MAP[CLASS_VOIDS]
        self.baseline_weight = baseline_weight
        
        # Initialize the selected baseline loss module
        self.base_loss = baseline if baseline is not None else nn.CrossEntropyLoss()
            
    def forward(self, logits, targets):
        # 1. Baseline Calculation
        base_loss = self.base_loss(logits, targets)
        
        # 2. Physics-Informed Penalty
        probs = F.softmax(logits, dim=1)
        void_probs = probs[:, self.void_class_index, :, :]
        
        # Calculate spatial gradients to detect predicted void perimeters
        dy = torch.abs(void_probs[:, 1:, :] - void_probs[:, :-1, :])
        dx = torch.abs(void_probs[:, :, 1:] - void_probs[:, :, :-1])
        
        # Sum of gradients approximates the perimeter; sum of probabilities approximates the area
        perimeter = torch.sum(dx) + torch.sum(dy)
        area = torch.sum(void_probs) + 1e-6 
        
        # Morphological penalty: Ratio of perimeter to area (penalizes fragmentation)
        fragmentation_penalty = perimeter / area
        
        total_loss = (self.baseline_weight * base_loss) + (self.physics_weight * fragmentation_penalty)
        
        return total_loss

    def to_dict(self):
        return {
            "baseline": self.base_loss.__class__.__name__,
            "baseline_weight": self.baseline_weight,
            "physics_weight": self.physics_weight
        }

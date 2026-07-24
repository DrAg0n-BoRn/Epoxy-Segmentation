import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedCompositeLoss(nn.Module):
    """
    A loss function that combines standard cross-entropy with a physics-informed penalty based on the predicted void class. 
    The penalty encourages the model to produce more contiguous void regions by penalizing fragmented predictions.
    """
    def __init__(self, ce_weight=1.0, physics_weight=0.1, void_class_index=2):
        """
        Args:
            ce_weight (float): Weight for the cross-entropy loss component.
            physics_weight (float): Weight for the physics-informed penalty component.
            void_class_index (int): The index of the void class in the logits tensor.
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.physics_weight = physics_weight
        self.void_class_index = void_class_index
        
    def forward(self, logits, targets):
        # Standard Cross-Entropy calculation
        base_loss = self.ce_loss(logits, targets)
        
        # Convert raw logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Extract the probability map specifically for the Voids class
        void_probs = probs[:, self.void_class_index, :, :]
        
        # Calculate spatial gradients to detect predicted void perimeters
        dy = torch.abs(void_probs[:, 1:, :] - void_probs[:, :-1, :])
        dx = torch.abs(void_probs[:, :, 1:] - void_probs[:, :, :-1])
        
        # Sum of gradients approximates the perimeter; sum of probabilities approximates the area
        perimeter = torch.sum(dx) + torch.sum(dy)
        area = torch.sum(void_probs) + 1e-6 
        
        # Morphological penalty: Ratio of perimeter to area (penalizes fragmentation)
        fragmentation_penalty = perimeter / area
        
        # Aggregate the final loss
        total_loss = (self.ce_weight * base_loss) + (self.physics_weight * fragmentation_penalty)
        
        return total_loss

    def to_dict(self):
        return {
            "ce_weight": self.ce_weight,
            "physics_weight": self.physics_weight,
            "void_class_index": self.void_class_index
        }

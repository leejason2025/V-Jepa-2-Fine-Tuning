"""Training losses for V-JEPA2-AC predictor."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VJEPA2Loss(nn.Module):
    """Simplified loss for V-JEPA2-AC predictor.

    Implements L1 distance between predicted and target representations.
    """

    def __init__(
        self,
        teacher_forcing_steps: int = 15,
        rollout_steps: int = 2,
        weight_tf: float = 1.0,
        weight_rollout: float = 1.0,
    ):
        """Initialize V-JEPA2 loss.

        Args:
            teacher_forcing_steps: Not used (for compatibility)
            rollout_steps: Not used (for compatibility)
            weight_tf: Weight for teacher-forcing loss (not used)
            weight_rollout: Weight for rollout loss (not used)
        """
        super().__init__()
        # Store for compatibility but currently using simple L1 loss
        self.teacher_forcing_steps = teacher_forcing_steps
        self.rollout_steps = rollout_steps
        self.weight_tf = weight_tf
        self.weight_rollout = weight_rollout

    def teacher_forcing_loss(
        self,
        predictor: nn.Module,
        encoder_features: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute teacher-forcing loss over T timesteps.

        At each timestep t, predict features at t+1 given:
        - Encoder features from 0 to t
        - Actions from 0 to t
        - States from 0 to t

        Args:
            predictor: Predictor model
            encoder_features: [B, T, H, W, D] encoder representations
            actions: [B, T, action_dim] action sequences
            states: [B, T, state_dim] state sequences

        Returns:
            L1 loss averaged over all timesteps
        """
        B, T, H, W, D = encoder_features.shape

        total_loss = 0.0
        num_predictions = 0

        # Teacher-forcing: predict each next step independently
        for t in range(min(T - 1, self.teacher_forcing_steps)):
            # Input: frames 0 to t
            input_features = encoder_features[:, :t+1]
            input_actions = actions[:, :t+1]
            input_states = states[:, :t+1]

            # Predict: frame t+1
            predicted = predictor(
                input_features,
                input_actions,
                input_states,
                predict_steps=1
            )  # [B, 1, H, W, D]

            # Target: actual frame t+1
            target = encoder_features[:, t+1:t+2]  # [B, 1, H, W, D]

            # L1 loss
            loss = F.l1_loss(predicted, target, reduction='mean')
            total_loss += loss
            num_predictions += 1

        # Average over all predictions
        if num_predictions > 0:
            total_loss = total_loss / num_predictions

        return total_loss

    def rollout_loss(
        self,
        predictor: nn.Module,
        encoder_features: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rollout loss for multi-step prediction.

        Predict multiple steps ahead by feeding predictor outputs back as inputs.
        Example for T=2:
            - Input: z_1, action_1, state_1
            - Predict: z_2_pred
            - Input: z_2_pred, action_2, state_2
            - Predict: z_3_pred
            - Loss: ||z_3_pred - z_3_target||_1

        Args:
            predictor: Predictor model
            encoder_features: [B, T, H, W, D] encoder representations
            actions: [B, T, action_dim] action sequences
            states: [B, T, state_dim] state sequences

        Returns:
            L1 loss for final rollout prediction
        """
        B, T, H, W, D = encoder_features.shape

        # Start from first frame
        current_features = encoder_features[:, 0:1]  # [B, 1, H, W, D]

        # Rollout for specified number of steps
        for step in range(self.rollout_steps):
            # Get action and state for this step
            current_action = actions[:, step:step+1]  # [B, 1, action_dim]
            current_state = states[:, step:step+1]    # [B, 1, state_dim]

            # Predict next frame
            predicted = predictor(
                current_features,
                current_action,
                current_state,
                predict_steps=1
            )  # [B, 1, H, W, D]

            # Use prediction as input for next step
            current_features = torch.cat([current_features, predicted], dim=1)

        # Final prediction
        final_prediction = current_features[:, -1:]  # [B, 1, H, W, D]

        # Target: actual frame at rollout_steps + 1
        target_idx = self.rollout_steps
        if target_idx < T:
            target = encoder_features[:, target_idx:target_idx+1]  # [B, 1, H, W, D]

            # L1 loss
            loss = F.l1_loss(final_prediction, target, reduction='mean')
        else:
            # Not enough frames for rollout, return zero loss
            loss = torch.tensor(0.0, device=encoder_features.device)

        return loss

    def forward(
        self,
        predictor: nn.Module,
        encoder_features: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute prediction loss.

        Args:
            predictor: Predictor model
            encoder_features: [B, N, D] encoder representations
            actions: [B, T, action_dim] action sequences
            states: [B, T, state_dim] state sequences

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # Run predictor
        predictions = predictor(encoder_features, actions, states)

        # Compute L1 loss between predictions and features
        loss = F.l1_loss(predictions, encoder_features, reduction='mean')

        # Loss dictionary for logging
        loss_dict = {
            'total_loss': loss.item(),
            'l1_loss': loss.item(),
        }

        return loss, loss_dict


def create_loss_function(config) -> VJEPA2Loss:
    """Create loss function from configuration.

    Args:
        config: Configuration object

    Returns:
        VJEPA2Loss instance
    """
    return VJEPA2Loss(
        teacher_forcing_steps=config.training.teacher_forcing_steps,
        rollout_steps=config.training.rollout_steps,
        weight_tf=config.training.loss_weight_tf,
        weight_rollout=config.training.loss_weight_rollout,
    )

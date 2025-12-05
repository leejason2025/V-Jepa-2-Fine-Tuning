"""
Local DROID-100 dataset loader for PyTorch (loads from downloaded pickle files).
"""

import os
import io
import pickle
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class LocalDROIDDataset(Dataset):
    """Load DROID-100 from local pickle files."""

    def __init__(
        self,
        data_dir="/workspace/V-Jepa-2-Fine-Tuning/data/droid_100",
        frames_per_clip=8,
        crop_size=256,
        transform=None,
    ):
        self.data_dir = data_dir
        self.frames_per_clip = frames_per_clip
        self.crop_size = crop_size

        # Get all episode files
        self.episode_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.pkl')
        ])

        if len(self.episode_files) == 0:
            raise ValueError(f"No episode files found in {data_dir}")

        print(f"Found {len(self.episode_files)} episodes")

        # Image transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(int(crop_size * 1.15)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Build index of all valid clips
        self.clips = []
        for ep_file in self.episode_files:
            with open(ep_file, 'rb') as f:
                episode = pickle.load(f)
            num_frames = len(episode['images_bytes'])
            # Create clips with stride
            for start_idx in range(0, num_frames - frames_per_clip + 1, 4):
                self.clips.append((ep_file, start_idx))

        print(f"Total clips: {len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def poses_to_diffs(self, poses):
        """Convert poses to action deltas."""
        xyz = poses[:, :3]
        thetas = poses[:, 3:6]
        gripper = poses[:, -1:]

        # Position deltas
        xyz_diff = xyz[1:] - xyz[:-1]

        # Rotation deltas
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack(angle_diff, axis=0)

        # Gripper deltas
        gripper_delta = gripper[1:] - gripper[:-1]

        return np.concatenate([xyz_diff, angle_diff, gripper_delta], axis=1)

    def __getitem__(self, idx):
        ep_file, start_idx = self.clips[idx]

        # Load episode
        with open(ep_file, 'rb') as f:
            episode = pickle.load(f)

        # Extract clip
        end_idx = start_idx + self.frames_per_clip
        clip_images_bytes = episode['images_bytes'][start_idx:end_idx]

        # Reshape cartesian_position from flat array to (num_frames, 6)
        num_frames_total = len(episode['images_bytes'])
        cart_pos_full = episode['cartesian_position'].reshape(num_frames_total, 6)
        grip_pos_full = episode['gripper_position'].reshape(num_frames_total, 1)

        clip_cart_pos = cart_pos_full[start_idx:end_idx]
        clip_grip_pos = grip_pos_full[start_idx:end_idx]

        # Decode images
        images = []
        for img_bytes in clip_images_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            img_tensor = self.transform(img)
            images.append(img_tensor)

        images = torch.stack(images)  # [T, C, H, W]

        # Reconstruct states: cart_pos (6D) + gripper (1D) = 7D
        states = np.concatenate([clip_cart_pos, clip_grip_pos], axis=1)  # [T, 7]

        # Compute action deltas (T-1 actions for T frames)
        actions = self.poses_to_diffs(states)  # [T-1, 7]

        # Convert to V-JEPA2 format: [T, H, W, C]
        images_nhwc = images.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]

        # Convert to torch
        actions_torch = torch.from_numpy(actions).float()
        states_torch = torch.from_numpy(states).float()

        return images_nhwc, actions_torch, states_torch


def create_dataloader(batch_size=2, num_workers=0, **kwargs):
    """Create dataloader for local DROID-100."""
    dataset = LocalDROIDDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader

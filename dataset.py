from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import data_augment as augment
import mediapipe as mp
pose_edges = list(mp.solutions.pose.POSE_CONNECTIONS)
hand_edges = list(mp.solutions.hands.HAND_CONNECTIONS)

# Offsets
POSE_OFFSET = 0
LEFT_HAND_OFFSET = 6
RIGHT_HAND_OFFSET = 6 + 21

edges = []
# Pose edges
for i, j in pose_edges:
    edges.append((i + POSE_OFFSET, j + POSE_OFFSET))
# Left hand edges
for i, j in hand_edges:
    edges.append((i + LEFT_HAND_OFFSET, j + LEFT_HAND_OFFSET))
# Right hand edges
for i, j in hand_edges:
    edges.append((i + RIGHT_HAND_OFFSET, j + RIGHT_HAND_OFFSET))
edges.append((15, LEFT_HAND_OFFSET + 0))
edges.append((16, RIGHT_HAND_OFFSET + 0))

V = 75
A = np.zeros((V, V))

for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1  # undirected

print("IN DATASET --------------------")

class ASLDataset(Dataset):
    def __init__(self, asl_df, gloss_to_idx, model_type):
        self.sample_list = []
        self.label_list = []
        # max_frames = augment.find_max_frames(asl_df)
        max_frames = 30
        for i, row in asl_df.iterrows():
            gloss = row['ASL_GLOSSES']
            # gloss = asl_gloss[:-2]
            if gloss not in gloss_to_idx:
                    raise ValueError(f"Unknown gloss: {gloss}")
            label_id = gloss_to_idx[gloss]
            pose, hands = np.array(row['ASL_POSES']), np.array(row['ASL_HANDS'])
            pose, hands = pose[:, :, 0:3], hands[:, :, 0:3]
            # # Regular
            # flattened = augment.flatten_landmarks(pose, hands, max_frames, model_type)
            # self.sample_list.append(flattened) # the first 33*4 are pose and the last 21*4 are hand
            # self.label_list.append(label_id)
            # Skip frames
            for k in [2]:
                pose_skipped, hands_skipped = augment.skip_frames(pose, k), augment.skip_frames(hands, k)
                flattened_skipped = augment.flatten_landmarks(pose_skipped, hands_skipped, max_frames, model_type)
                self.sample_list.append(flattened_skipped) # the first 33*4 are pose and the last 21*4 are hand
                self.label_list.append(label_id)
            # # Add frames
            # pose_added, hands_added = augment.add_frames(pose), augment.add_frames(hands)
            # flattened_added = augment.flatten_landmarks(pose_added, hands_added, max_frames, model_type)
            # self.sample_list.append(flattened_added) # the first 33*4 are pose and the last 21*4 are hand
            # self.label_list.append(label_id)

            # # Mirror poses
            # pose_mirrored, hands_mirrored = augment.mirror_coords(pose), augment.mirror_coords(hands)
            # flattened_mirrored = augment.flatten_landmarks(pose_mirrored, hands_mirrored, max_frames, model_type)
            # self.sample_list.append(flattened_mirrored) # the first 33*4 are pose and the last 21*4 are hand
            # self.label_list.append(label_id)

            # # Gaussian noise
            # pose_noisy, hands_noisy = augment.add_gaussian_noise(pose), augment.add_gaussian_noise(hands)
            # flattened_noisy = augment.flatten_landmarks(pose_noisy, hands_noisy, max_frames, model_type)
            # self.sample_list.append(flattened_noisy) # the first 33*4 are pose and the last 21*4 are hand
            # self.label_list.append(label_id)

            # Rotate hands around wrist
            hands_rotated = augment.augment_hands(hands, 5, 5, 10)
            flattened_rotated = augment.flatten_landmarks(pose, hands_rotated, max_frames, model_type)
            self.sample_list.append(flattened_rotated) # the first 33*4 are pose and the last 21*4 are hand
            self.label_list.append(label_id)
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        label = self.label_list[idx]
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label)
        return sample, label


class ASLTestDataset(Dataset):
    def __init__(self, asl_df, gloss_to_idx, model_type):
        self.sample_list = []
        self.label_list = []
        max_frames = 30
        for i, row in asl_df.iterrows():
            gloss = row['ASL_GLOSSES']
            # gloss = asl_gloss[:-2]
            if gloss in gloss_to_idx:
                label_id = gloss_to_idx[gloss]
                pose, hands = np.array(row['ASL_POSES']), np.array(row['ASL_HANDS'])

                # Regular
                flattened = augment.flatten_landmarks(pose, hands, max_frames, model_type)
                self.sample_list.append(flattened) # the first 33*4 are pose and the last 21*4 are hand
                self.label_list.append(label_id)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        label = self.label_list[idx]
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label)
        return sample, label

# C, T, V: Coordinates, Number of Frames, Number of Joints
class ISLDataset(Dataset):
    def __init__(self, isl_df, gloss_to_idx, model_type):
        self.sample_list = []
        self.label_list = []
        max_frames = 30
        for i, row in isl_df.iterrows():
            gloss = row['ISL_GLOSSES']
            if gloss in gloss_to_idx:
                label_id = gloss_to_idx[gloss]
                pose, hands = np.array(row['ISL_POSES']), np.array(row['ISL_HANDS'])
                pose, hands = pose[:, :, 0:3], hands[:, :, 0:3]
                # Regular
                flattened = augment.flatten_landmarks(pose, hands, max_frames, model_type)
                self.sample_list.append(flattened)
                self.label_list.append(label_id)
                
                # Skip frames
                for k in [2]:
                    pose_skipped, hands_skipped = augment.skip_frames(pose, k), augment.skip_frames(hands, k)
                    flattened_skipped = augment.flatten_landmarks(pose_skipped, hands_skipped, max_frames, model_type)
                    self.sample_list.append(flattened_skipped) # the first 33*4 are pose and the last 21*4 are hand
                    self.label_list.append(label_id)
                # # Add frames
                # pose_added, hands_added = augment.add_frames(pose), augment.add_frames(hands)
                # flattened_added = augment.flatten_landmarks(pose_added, hands_added, max_frames, model_type)
                # self.sample_list.append(flattened_added) # the first 33*4 are pose and the last 21*4 are hand
                # self.label_list.append(label_id)

                # # Mirror poses
                # pose_mirrored, hands_mirrored = augment.mirror_coords(pose), augment.mirror_coords(hands)
                # flattened_mirrored = augment.flatten_landmarks(pose_mirrored, hands_mirrored, max_frames, model_type)
                # self.sample_list.append(flattened_mirrored) # the first 33*4 are pose and the last 21*4 are hand
                # self.label_list.append(label_id)

                # # Gaussian noise
                # pose_noisy, hands_noisy = augment.add_gaussian_noise(pose), augment.add_gaussian_noise(hands)
                # flattened_noisy = augment.flatten_landmarks(pose_noisy, hands_noisy, max_frames, model_type)
                # self.sample_list.append(flattened_noisy) # the first 33*4 are pose and the last 21*4 are hand
                # self.label_list.append(label_id)

                # Rotate hands around wrist
                hands_rotated = augment.augment_hands(hands, 5, 5, 10)
                flattened_rotated = augment.flatten_landmarks(pose, hands_rotated, max_frames, model_type)
                self.sample_list.append(flattened_rotated) # the first 33*4 are pose and the last 21*4 are hand
                self.label_list.append(label_id)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        label = self.label_list[idx]
        sample = torch.from_numpy(sample)
        label = torch.tensor(label)
        return sample, label

class ISLTestDataset(Dataset):
    def __init__(self, isl_df, gloss_to_idx, model_type):
        self.sample_list = []
        self.label_list = []
        max_frames = 30
        for i, row in isl_df.iterrows():
            gloss = row['ISL_GLOSSES']
            if gloss in gloss_to_idx:
                label_id = gloss_to_idx[gloss]
                pose, hands = np.array(row['ISL_POSES']), np.array(row['ISL_HANDS'])
                hands = hands.reshape(hands.shape[0], -1, 4)
                flattened = augment.flatten_landmarks(pose, hands, max_frames, model_type)
                self.sample_list.append(flattened)
                self.label_list.append(label_id)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        label = self.label_list[idx]
        sample = torch.from_numpy(sample)
        label = torch.tensor(label)
        return sample, label

class PairedDataset(Dataset):
    def __init__(self, isl_df, asl_df, shared_glosses, gloss_to_idx, model_type='T'):
        self.pairs = []
        self.labels = []
        max_frames = 30

        for gloss in shared_glosses:
            label_id = gloss_to_idx[gloss]
            isl_rows = isl_df[isl_df['ISL_GLOSSES'] == gloss]
            asl_rows = asl_df[asl_df['ASL_GLOSSES'] == gloss]
            isl_sequences = []
            for _, row in isl_rows.iterrows():
                pose, hands = np.array(row['ISL_POSES']), np.array(row['ISL_HANDS'])  
                flattened = augment.flatten_landmarks(pose, hands, max_frames, model_type)
                isl_sequences.append(flattened)

            asl_sequences = []
            for _, row in asl_rows.iterrows():
                pose, hands = np.array(row['ASL_POSES']), np.array(row['ASL_HANDS'])  
                flattened = augment.flatten_landmarks(pose, hands, max_frames, model_type)
                asl_sequences.append(flattened)
            # Pair every ISL sample with every ASL sample for this gloss
            max_pairs_per_gloss = 100
            pairs_for_gloss = []
            for isl_seq in isl_sequences:
                for asl_seq in asl_sequences:
                    pairs_for_gloss.append((isl_seq, asl_seq))
            
            # Randomly sample if over the cap
            if len(pairs_for_gloss) > max_pairs_per_gloss:
                indices = np.random.choice(len(pairs_for_gloss), max_pairs_per_gloss, replace=False)
                pairs_for_gloss = [pairs_for_gloss[i] for i in indices]
            for pair in pairs_for_gloss:
                self.pairs.append(pair)
                self.labels.append(label_id)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        isl_seq, asl_seq = self.pairs[idx]
        label = self.labels[idx]
        isl = torch.from_numpy(isl_seq).float()
        asl = torch.from_numpy(asl_seq).float()
        label = torch.tensor(label)
        return isl, asl, label

class TestDataset(Dataset):
    def __init__(self, isl_df, asl_df, shared_glosses, gloss_to_idx, model_type='T'):
        self.isl_samples, self.isl_labels = [], []
        self.asl_samples, self.asl_labels = [], []
        max_frames = 30

        for gloss in shared_glosses:
            label_id = gloss_to_idx[gloss]

            for _, row in isl_df[isl_df['ISL_GLOSSES'] == gloss].iterrows():
                pose, hands = np.array(row['ISL_POSES']), np.array(row['ISL_HANDS'])
                self.isl_samples.append(augment.flatten_landmarks(pose, hands, max_frames, model_type))
                self.isl_labels.append(label_id)

            for _, row in asl_df[asl_df['ASL_GLOSSES'] == gloss].iterrows():
                pose, hands = np.array(row['ASL_POSES']), np.array(row['ASL_HANDS'])
                self.asl_samples.append(augment.flatten_landmarks(pose, hands, max_frames, model_type))
                self.asl_labels.append(label_id)

    def get_isl(self):
        return (
            torch.tensor(np.array(self.isl_samples), dtype=torch.float32),
            torch.tensor(self.isl_labels, dtype=torch.long)
        )

    def get_asl(self):
        return (
            torch.tensor(np.array(self.asl_samples), dtype=torch.float32),
            torch.tensor(self.asl_labels, dtype=torch.long)
        )

    def __len__(self):
        return len(self.isl_samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.isl_samples[idx], dtype=torch.float32),
            torch.tensor(self.asl_samples[idx], dtype=torch.float32),
            torch.tensor(self.isl_labels[idx], dtype=torch.long)
        )
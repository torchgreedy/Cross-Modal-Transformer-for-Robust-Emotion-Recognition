import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

class MultimodalDataset(Dataset):
    def __init__(self, eeg, eye, eeg_mask, eye_mask, labels, groups=None):
        self.eeg_data = torch.FloatTensor(eeg)
        self.eye_data = torch.FloatTensor(eye)
        self.eeg_mask = torch.FloatTensor(eeg_mask)
        self.eye_mask = torch.FloatTensor(eye_mask)
        self.labels = torch.LongTensor(labels)
        self.groups = torch.LongTensor(groups) if groups is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'eeg': self.eeg_data[idx],
            'eye': self.eye_data[idx],
            'eeg_mask': self.eeg_mask[idx],
            'eye_mask': self.eye_mask[idx],
            'label': self.labels[idx]
        }
        if self.groups is not None:
            item['group'] = self.groups[idx]
        return item

def load_and_pad_subject(data_dir, subject_id, max_length=74, n_trials=45, eeg_features=310, eye_features=33):
    """Load and pad EEG and eye movement features for a single subject."""
    # Load EEG features
    eeg_data = np.load(os.path.join(data_dir, 'EEG_DE_features', f'{subject_id}_123.npz'))
    eeg_features_dict = pickle.loads(eeg_data['data'])
    eeg_labels_dict = pickle.loads(eeg_data['label'])

    # Load eye movement features
    eye_data = np.load(os.path.join(data_dir, 'Eye_movement_features', f'{subject_id}_123.npz'))
    eye_features_dict = pickle.loads(eye_data['data'])
    eye_labels_dict = pickle.loads(eye_data['label'])

    # Initialize padded arrays and masks
    padded_eeg = np.zeros((n_trials, max_length, eeg_features))
    padded_eye = np.zeros((n_trials, max_length, eye_features))
    eeg_mask = np.zeros((n_trials, max_length))
    eye_mask = np.zeros((n_trials, max_length))
    labels = np.zeros(n_trials, dtype=np.int64)

    # Fill arrays and masks
    for i in range(n_trials):
        eeg_seq = eeg_features_dict[i]
        eye_seq = eye_features_dict[i]
        eeg_seq_len = eeg_seq.shape[0]
        eye_seq_len = eye_seq.shape[0]

        padded_eeg[i, :eeg_seq_len, :] = eeg_seq
        padded_eye[i, :eye_seq_len, :] = eye_seq
        eeg_mask[i, :eeg_seq_len] = 1
        eye_mask[i, :eye_seq_len] = 1

        # Check label consistency
        assert eeg_labels_dict[i][0] == eye_labels_dict[i][0], f"Label mismatch in trial {i} for subject {subject_id}"
        labels[i] = int(eeg_labels_dict[i][0])

    return {
        'eeg': padded_eeg,
        'eye': padded_eye,
        'eeg_mask': eeg_mask,
        'eye_mask': eye_mask,
        'labels': labels
    }

def collect_all_subjects(data_dir, subject_ids):
    """Load and pad all subjects' data, return combined arrays and group labels."""
    all_eeg, all_eye, all_eeg_mask, all_eye_mask, all_labels, all_groups = [], [], [], [], [], []

    for idx, subject_id in enumerate(subject_ids):
        subj = load_and_pad_subject(data_dir, subject_id)
        all_eeg.append(subj['eeg'])
        all_eye.append(subj['eye'])
        all_eeg_mask.append(subj['eeg_mask'])
        all_eye_mask.append(subj['eye_mask'])
        all_labels.append(subj['labels'])
        all_groups.append(np.full(subj['labels'].shape, idx, dtype=np.int64))

    # Concatenate
    eeg = np.concatenate(all_eeg, axis=0)
    eye = np.concatenate(all_eye, axis=0)
    eeg_mask = np.concatenate(all_eeg_mask, axis=0)
    eye_mask = np.concatenate(all_eye_mask, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    return eeg, eye, eeg_mask, eye_mask, labels, groups

def get_logo_dataloaders(data_dir, batch_size=32, val_split=0.1):
    """Generator that yields (train_loader, val_loader, test_loader, fold_info) for each LOGO fold."""
    subject_ids = list(range(1, 17))
    eeg, eye, eeg_mask, eye_mask, labels, groups = collect_all_subjects(data_dir, subject_ids)

    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(eeg, labels, groups)):
        # Split train_idx into train/val
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_split, stratify=labels[train_idx], random_state=42
        )

        # Compute normalization on train only
        train_eeg, eeg_mean, eeg_std = normalize_features(eeg[train_idx], eeg_mask[train_idx])
        train_eye, eye_mean, eye_std = normalize_features(eye[train_idx], eye_mask[train_idx])

        # Apply normalization to val and test
        val_eeg = apply_normalization(eeg[val_idx], eeg_mean, eeg_std)
        val_eye = apply_normalization(eye[val_idx], eye_mean, eye_std)
        test_eeg = apply_normalization(eeg[test_idx], eeg_mean, eeg_std)
        test_eye = apply_normalization(eye[test_idx], eye_mean, eye_std)

        # Build datasets
        train_set = MultimodalDataset(
            train_eeg, train_eye, eeg_mask[train_idx], eye_mask[train_idx],
            labels[train_idx], groups[train_idx]
        )
        val_set = MultimodalDataset(
            val_eeg, val_eye, eeg_mask[val_idx], eye_mask[val_idx],
            labels[val_idx], groups[val_idx]
        )
        test_set = MultimodalDataset(
            test_eeg, test_eye, eeg_mask[test_idx], eye_mask[test_idx],
            labels[test_idx], groups[test_idx]
        )

        # Build dataloaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        fold_info = {
            'fold': fold + 1,
            'test_subject': subject_ids[groups[test_idx][0]],
            'train_subjects': [subject_ids[g] for g in np.unique(groups[train_idx])],
            'val_subjects': [subject_ids[g] for g in np.unique(groups[val_idx])],
        }

        yield train_loader, val_loader, test_loader, fold_info

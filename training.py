import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import gzip
from collections import defaultdict
import itertools
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

AA = "ACDEFGHIKLMNPQRSTVWYX"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
PAD_TOKEN_ID = len(AA)
MASK_TOKEN_ID = AA_INDEX['X']

def integer_encode(sequence):
    """Converts a protein sequence into a list of integer tokens."""
    return [AA_INDEX.get(aa, AA_INDEX['X']) for aa in sequence]

def integer_decode(encoded):
    """Converts a list of integer tokens back to a protein sequence."""
    id_to_aa = {i: aa for aa, i in AA_INDEX.items()}
    return "".join([id_to_aa.get(i, '?') for i in encoded])

def parse_uniref50_fasta(filepath, limit=None):
    """
    Parses a gzipped UniRef50 FASTA file, correctly handling multi-line sequences.
    Filters out sequences longer than 550 amino acids or not starting with Methionine (M).
    
    Args:
        filepath (str): Path to the uniref50.fasta.gz file.
        limit (int, optional): The number of sequences to read. If None, reads all sequences.

    Returns:
        list: List of (header, sequence) tuples.
    """
    data = []
    
    with gzip.open(filepath, 'rt') as f:
        header = None
        sequence_parts = []
        
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith('>'):
                if header is not None:
                    full_sequence = "".join(sequence_parts)
                    if len(full_sequence) <= 550 and len(full_sequence) > 0 and full_sequence.startswith('M'):
                         data.append((header, full_sequence))
                         if limit is not None and len(data) >= limit:
                             return data
                
                header = line
                sequence_parts = []
            else:
                sequence_parts.append(line)
        
        if header is not None and (limit is None or len(data) < limit):
            full_sequence = "".join(sequence_parts)
            if len(full_sequence) <= 550 and len(full_sequence) > 0 and full_sequence.startswith('M'):
                data.append((header, full_sequence))
                
    return data

def augment_sequence(encoded_seq, mask_prob=0.15):
    """
    Applies random masking to the sequence.
    """
    seq_len = len(encoded_seq)
    mask = np.random.rand(seq_len) < mask_prob
    augmented = np.copy(encoded_seq)
    augmented[mask] = MASK_TOKEN_ID
    return augmented

class ProteinDataset(Dataset):
    def __init__(self, data, max_len=512, transform=True):
        """
        Args:
            data (list): List of (header, sequence) tuples or just sequences.
            max_len (int): Maximum sequence length.
            transform (bool): Whether to apply augmentation.
        """
        self.data = data
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if isinstance(item, tuple):
            seq = item[1]
        else:
            seq = item
            
        encoded_seq = np.array(integer_encode(seq))
        
        def process_seq(s):
            if len(s) > self.max_len:
                s = s[:self.max_len]
            padded = np.full(self.max_len, PAD_TOKEN_ID, dtype=np.int64)
            padded[:len(s)] = s
            return padded

        if self.transform:
            view1 = augment_sequence(encoded_seq)
            view2 = augment_sequence(encoded_seq)
            return torch.tensor(process_seq(view1)), torch.tensor(process_seq(view2))
        else:
            return torch.tensor(process_seq(encoded_seq)), torch.tensor(process_seq(encoded_seq))


class ProteinConvNetEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=256, filter_sizes=[3, 5, 7], output_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_TOKEN_ID)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = torch.cat(pooled, dim=1)
        
        embedding_vector = self.fc(cat)
        embedding_vector = F.normalize(embedding_vector, p=2, dim=1)
        
        return embedding_vector


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot provide both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # log probs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9) # avoidss div by zero

        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# TODO: 
#       - Change temperature to a hyperparameter
#       - Add a learning rate scheduler
#       - Add a checkpointing func
#       - Add a early stopping mechanism (simple loop?)
#       - Add L2 regularization? Needed?

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} Batches", leave=False)
    for view1, view2 in pbar:
        
        view1, view2 = view1.to(device), view2.to(device)
        bsz = view1.shape[0]
        
        optimizer.zero_grad()
        
        combined_input = torch.cat([view1, view2], dim=0)
        combined_features = model(combined_input) # [2*bsz, output_dim]
        
        # Reshape for loss function: [bsz, 2, output_dim]
        f1, f2 = torch.split(combined_features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
        
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set for one epoch.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for view1, view2 in dataloader:
            view1, view2 = view1.to(device), view2.to(device)
            bsz = view1.shape[0]
            
            combined_input = torch.cat([view1, view2], dim=0)
            combined_features = model(combined_input)
            
            f1, f2 = torch.split(combined_features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            loss = criterion(features)
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def get_embedding(model, sequence, max_len, device):
    """
    Generates an embedding for a single protein sequence using the trained model.

    Args:
        model (nn.Module): The trained ProteinConvNetEncoder model.
        sequence (str): The protein sequence to embed.
        max_len (int): The maximum sequence length the model was trained with.
        device (torch.device): The device the model is on (e.g., 'cpu', 'mps').

    Returns:
        np.ndarray: The embedding vector for the sequence.
    """
    model.eval() 
    
    encoded_seq = integer_encode(sequence)
    if len(encoded_seq) > max_len:
        encoded_seq = encoded_seq[:max_len]
    
    padded_seq = np.full(max_len, PAD_TOKEN_ID, dtype=np.int64)
    padded_seq[:len(encoded_seq)] = encoded_seq
    
    seq_tensor = torch.tensor(padded_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(seq_tensor)
        
    return embedding.cpu().numpy()


def main():
    UNIREF_PATH = './uniref50/enzymes.fasta.gz'
    DATA_LIMIT = None
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 128
    EMBEDDING_DIM = 128
    OUTPUT_DIM = 256
    NUM_FILTERS = 256
    FILTER_SIZES = [3, 5, 9, 15]
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
         device = torch.device("cuda")
    
    print(f"Using device: {device}")
    data = parse_uniref50_fasta(UNIREF_PATH, limit=DATA_LIMIT)
    print(f"{len(data)} sequences.")
    
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42
    )

    print(f"Training on {len(train_data)} sequences, validating on {len(val_data)} sequences.")

    # Transform=True for training to apply augmentation
    train_dataset = ProteinDataset(train_data, max_len=MAX_SEQ_LEN, transform=True)
    val_dataset = ProteinDataset(val_data, max_len=MAX_SEQ_LEN, transform=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ProteinConvNetEncoder(
        vocab_size=len(AA) + 1,  # +1 PAD_TOKEN
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        output_dim=OUTPUT_DIM
    ).to(device)
    
    criterion = SupConLoss(temperature=0.1)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training (SimCLR - Self-Supervised)...")
    epoch_pbar = tqdm(range(EPOCHS), desc="Epochs")
    for epoch in epoch_pbar:
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch, EPOCHS)
        avg_val_loss = validate_epoch(model, val_dataloader, criterion, device)
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}'
        })
        
    torch.save(model.state_dict(), 'model.pth')
    print("Training complete. Model saved to 'model.pth'")


if __name__ == '__main__':
    main()

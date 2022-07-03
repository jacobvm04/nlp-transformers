import numpy as np
import torch
import gzip

class Enwik8CharsDataset(torch.utils.data.Dataset):
    @staticmethod
    def collate(batch):
        input_seqs, target_seqs = zip(*batch)

        inputs = torch.cat([seq[None, :] for seq in input_seqs], dim=0).long()
        targets = torch.cat([seq[None, :] for seq in target_seqs], dim=0).long()

        return inputs, targets

    def __init__(self, context_len):
        self.context_len = context_len

        with gzip.open('datasets/enwik8.gz') as f:
            data = np.fromstring(f.read(), dtype=np.uint8)
            self.data = torch.from_numpy(data).long()

    def __len__(self):
        return self.data.size(0) - 2 * self.context_len

    def __getitem__(self, index):
        input_seq = self.data[index:index + self.context_len]
        target_seq = self.data[index + 1:index + self.context_len + 1]

        return input_seq, target_seq

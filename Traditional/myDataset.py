
import linecache
from torch.utils.data import Dataset
import numpy as np
    
class FileDataset(Dataset):
    def __init__(self, filename, vocab_file_name, max_uttr_num, max_uttr_len, max_response_len, dataset="douban"):
        super(FileDataset, self).__init__()
        self._filename = filename
        self._vocab_file_name = vocab_file_name
        self._max_uttr_num = max_uttr_num
        self._max_uttr_len = max_uttr_len
        self._max_response_len = max_response_len
        self._dataset = dataset
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
        self.vocab = self.load_vocab()
    
    def load_vocab(self):
        vocab = {"_PAD_": 0}
        with open(self._vocab_file_name, 'r', encoding='utf8') as f:
            for line in f:
                word = line.strip().split("\t")[0]
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        label = int(line[0])
        context = line[:-1]
        response = line[-1]
        utterances = context[-self._max_uttr_num:]
        us_vec = []
        us_mask = []
        # TODO: check pad last or pad first
        for utterance in utterances:
            u_tokens = utterance.split(' ')[:self._max_uttr_len]
            u_len = len(u_tokens)
            u_vec = [self.vocab.get(x, 1) for x in u_tokens]
            u_mask = [0] * len(u_vec)
            u_pad_len = self._max_uttr_len - u_len
            u_vec += [0] * u_pad_len
            u_mask += [1] * u_pad_len
            us_vec.append(u_vec)
            us_mask.append(u_mask)
        us_pad_num = self._max_uttr_num - len(utterances)
        us_pad_vec = [[0] * self._max_uttr_len] * us_pad_num
        us_pad_mask = [[1] * self._max_uttr_len] * us_pad_num
        us_vec = us_pad_vec + us_vec
        us_mask = us_pad_mask + us_mask

        r_tokens = response.split(' ')[:self._max_uttr_len]
        r_len = len(r_tokens)
        r_vec = [self.vocab.get(x, 1) for x in r_tokens]
        r_mask = [0] * len(r_vec)
        r_pad_len = self._max_uttr_len - r_len
        r_vec += [0] * r_pad_len
        r_mask += [1] * r_pad_len
        
        batch = {
            "ctx": np.asarray(us_vec),
            "rep": np.asarray(r_vec),
            "ctx_mask": np.asarray(us_mask),
            "rep_mask": np.asarray(r_mask),
            "labels": label
        }

        return batch
    
    def __len__(self):
        return self._total_data


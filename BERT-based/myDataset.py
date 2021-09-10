import linecache
from torch.utils.data import Dataset
import numpy as np

class FileDataset(Dataset):
    def __init__(self, filename, max_context_len, max_response_len, max_sequence_len, tokenizer):
        super(FileDataset, self).__init__()
        self._filename = filename
        self._max_context_len = max_context_len
        self._max_response_len = max_response_len
        self._max_seq_length = max_sequence_len
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    
    def annotate(self, context, response):
        all_context_sents = []
        for sent in context:
            sent = "".join(sent.split())
            sent = self._tokenizer.tokenize(sent)[:self._max_context_len]
            all_context_sents.append(sent)
        tokens = [self._tokenizer.cls_token]
        segment_ids = [0]
        for sent in all_context_sents:
            tokens.extend(sent + ["[eos]"])
            segment_ids.extend([1] * (len(sent) + 1))
        tokens += [self._tokenizer.sep_token]
        segment_ids += [0]
        
        response_tokens = self._tokenizer.tokenize(response)[:self._max_response_len]
        tokens += response_tokens
        segment_ids += [1] * len(response_tokens)
        tokens += [self._tokenizer.sep_token]
        segment_ids += [1]
        
        tokens = tokens[:self._max_seq_length]
        segment_ids = segment_ids[:self._max_seq_length]
        all_attention_mask = [1] * len(tokens)
        assert len(tokens) <= self._max_seq_length
        while len(tokens) < self._max_seq_length:
            tokens.append(self._tokenizer.pad_token)
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(tokens) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(tokens)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        label = int(line[0])
        context = line[:-1]
        response = line[-1]
        input_ids, attention_mask, segment_ids = self.annotate(context, response)
        batch = {
            'input_ids': input_ids, 
            'token_type_ids': segment_ids, 
            'attention_mask': attention_mask, 
            'labels': int(label),
        }
        return batch
    
    def __len__(self):
        return self._total_data


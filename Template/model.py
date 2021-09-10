
import torch
import torch.nn as nn
import torch.nn.init as init


class MyModel(nn.Module):
    def __init__(self, word_embeddings):
        super(MyModel, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        # init.xavier_normal_(self.cnn_2d_1.weight)
        # for weights in [self.gru_encoder.weight_hh_l0, self.gru_encoder.weight_ih_l0]:
        #     init.orthogonal_(weights)
        pass


    def forward(self, batch):
        pass
        
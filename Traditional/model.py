
import torch
import torch.nn as nn
import torch.nn.init as init


class MyModel(nn.Module):
    def __init__(self, word_embeddings):
        super(MyModel, self).__init__()
        gru_hidden = 200
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False, padding_idx=0)
        self.gru_encoder = nn.GRU(input_size=400, hidden_size=gru_hidden, batch_first=True)
        self.cnn_2d_1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3))
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.cnn_2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.cnn_2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.affine2 = nn.Linear(in_features=3 * 3 * 64, out_features=200)
        self.gru_acc = nn.GRU(input_size=200, hidden_size=gru_hidden, batch_first=True)
        self.affine_out = nn.Linear(in_features=gru_hidden, out_features=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.cnn_2d_1.weight)
        init.xavier_normal_(self.cnn_2d_2.weight)
        init.xavier_normal_(self.cnn_2d_3.weight)
        init.xavier_normal_(self.affine2.weight)
        init.xavier_normal_(self.affine_out.weight)
        for weights in [self.gru_encoder.weight_hh_l0, self.gru_encoder.weight_ih_l0]:
            init.orthogonal_(weights)
        for weights in [self.gru_acc.weight_hh_l0, self.gru_acc.weight_ih_l0]:
            init.orthogonal_(weights)
    
    def distance(self, A, B, C, epsilon=1e-6):
        M1 = torch.einsum("bud,dd,brd->bur", [A, B, C])
        A_norm = A.norm(dim=-1)
        C_norm = C.norm(dim=-1)
        M2 = torch.einsum("bud,brd->bur", [A, C]) / (torch.einsum("bu,br->bur", A_norm, C_norm) + epsilon)
        return M1, M2

    def get_Matching_Map(self, bU_embedding, bR_embedding):
        '''
        :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
        :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
        :return: E: (bsz*max_utterances, max_u_words, max_r_words)
        '''
        M1, M2 = self.distance(bU_embedding, self.A, bR_embedding)
        M = torch.stack([M1, M2], dim=1)
        return M

    def UR_Matching(self, bU_embedding, bR_embedding):
        '''
        :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
        :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
        :return: (bsz*max_utterances, (max_u_words - width)/stride + 1, (max_r_words -height)/stride + 1, channel)
        '''
        M = self.get_Matching_Map(bU_embedding, bR_embedding)
        Z = self.relu(self.cnn_2d_1(M))
        Z = self.maxpooling1(Z)
        Z = self.relu(self.cnn_2d_2(Z))
        Z = self.maxpooling2(Z)
        Z = self.relu(self.cnn_2d_3(Z))
        Z = self.maxpooling3(Z)
        Z = Z.view(Z.size(0), -1)  # (bsz*max_utterances, *)
        V = self.tanh(self.affine2(Z))   # (bsz*max_utterances, 50)
        return V

    def forward(self, batch):
        '''
        :param bU: batch utterance, size: (batch_size, max_utterances, max_u_words)
        :param bR: batch responses, size: (batch_size, max_r_words)
        :return: scores, size: (batch_size, )
        '''
        bU = batch["ctx"]
        bR = batch["rep"]
        bU_embedding = self.dropout(self.word_embedding(bU))
        bR_embedding = self.dropout(self.word_embedding(bR))
        bU_rep, _ = self.gru_encoder(bU_embedding)
        bR_rep, _ = self.gru_encoder(bR_embedding)
        su1, su2, su3, su4 = bU_rep.size()
        multi_context = bU_rep.view(-1, su3, su4)
        sr1, sr2, sr3 = bR_embedding.size()
        bR_rep = bR_rep.unsqueeze(dim=1).repeat(1, su2, 1, 1)
        bR_rep = bR_rep.view(-1, sr2, sr3)
        V = self.UR_Matching(multi_context, bR_rep)
        V = V.view(su1, su2, -1)
        H, _ = self.gru_acc(V)
        L = self.dropout(H[:, -1, :])
        output = self.affine_out(L)
        return output.squeeze(1)

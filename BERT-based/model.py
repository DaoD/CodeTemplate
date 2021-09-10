from transformers import BertModel
import torch.nn as nn
import torch.nn.init as init

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 1)
        self.dropout = nn.Dropout(0.1)
        init.xavier_normal_(self.classifier.weight)
    
    def forward(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        sent_rep = self.dropout(self.bert_model(**bert_inputs)[1])
        y_pred = self.classifier(self.relu(sent_rep))
        return y_pred.squeeze(1)

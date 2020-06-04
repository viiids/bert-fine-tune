import torch
from torch import nn
from transformers import BertModel


class TweetExtractorModel(nn.Module):
    def __init__(self, freeze_bert=True):
        super(TweetExtractorModel, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        # In first experiment, not training the previous layers
        # TODO(Viman): Before training on GPUs, remove this
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Final layer. Needs two outputs which are supposed to be logits: startIndex and endIndex
        self.dropout = nn.Dropout(0.2)
        # 768 because output is a vector of size 768 (embeddings size)
        self.fc = nn.Linear(768, 2)
        # Intialize the fc layer
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attn_masks):

        # Feeding the input to BERT model to obtain hidden_states of all the layers
        _, hidden_states = self.bert_layer(input_ids, attn_masks)

        hidden_states_at_each_layer = hidden_states[-1]
        # embeddings_output = hidden_states[-2]
        # X = torch.cat((hidden_states_at_each_layer, embeddings_output), dim=-1)
        X = self.dropout(hidden_states_at_each_layer)
        logits = self.fc(X)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

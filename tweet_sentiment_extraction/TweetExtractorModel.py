import torch
from torch import nn
from transformers import BertModel


class TweetExtractorModel(nn.Module):
    def __init__(self, freeze_bert=True):
        super(TweetExtractorModel, self).__init__()
        # Instantiating BERT model object
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # TODO(Viman): Before training on GPUs and finalization, remove this
        # Freeze bert layers
        # In first experiment, not training the previous layers
        if freeze_bert:
            for p in self.bert_model.parameters():
                p.requires_grad = False

        # Final layer. Needs two outputs which are supposed to be logits: startIndex and endIndex
        self.dropout = nn.Dropout(0.2)
        # 768 because output is a vector of size 768 (embeddings size)
        self.fc = nn.Linear(768, 2)
        # Intialize the fc layer
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attn_masks, token_type_ids):

        # Feeding the input to BERT model to obtain hidden_states of all the layers
        response = self.bert_model(input_ids, attn_masks, token_type_ids)

        hidden_states = response[0]

        # Shape of hidden_states is (1, 50, 768)
        # TODO(Viman): Try mean as opposed to max
        hidden_states, _ = torch.max(hidden_states, dim=1)

        X = self.dropout(hidden_states)
        logits = self.fc(X)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

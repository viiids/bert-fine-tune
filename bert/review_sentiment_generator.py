import torch
from transformers import BertModel
from transformers import BertTokenizer


def load_bert_model():
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    print("Loaded BERT base model")
    return bert_model


def get_bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_input(sentence):
    tokenizer = get_bert_tokenizer()
    tokens = tokenizer.tokenize(text=sentence)
    return tokens


def add_delimiter_tokens(tokens):
    return ['[CLS]'] + tokens + ['[SEP]']


def pad_input(tokens, max_len=50):
    padded_tokens = tokens + ['[PAD]' for _ in range(max_len - len(tokens))]
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    return padded_tokens, attn_mask


def create_segment_tokens(tokens, max_len=50):
    # TODO(viman): If the use cases requires two sequences, then this must set 1s for the second sequence (after [SEP])
    if tokens.count('[SEP]') > 1:
        raise ValueError('Not implemented as yet')
    seg_ids = [0 for _ in range(max_len)]
    return seg_ids

def execute():
    bert_model = load_bert_model()
    tokens = tokenize_input("This isn't my problem(s); at all.")
    tokens = add_delimiter_tokens(tokens)
    padded_tokens, attn_mask = pad_input(tokens)
    print(padded_tokens)
    seg_ids = create_segment_tokens(tokens)
    token_ids = get_bert_tokenizer().convert_tokens_to_ids(padded_tokens)
    print(token_ids)

    token_ids = torch.tensor(token_ids).unsqueeze(0)  # Shape : [1, max_len]
    attn_mask = torch.tensor(attn_mask).unsqueeze(0)  # Shape : [1, max_len]
    seg_ids = torch.tensor(seg_ids).unsqueeze(0)  # Shape : [1, max_len]

    print(token_ids.shape)
    print(attn_mask.shape)
    print(seg_ids.shape)

    # Bert model returns two outpus, hidden_reps is the hidden states of each token after passing them through the
    # series of self attention layers. cls_head contains the final output of the encoder after passing [CLS] token
    # through the fully connected layer with tanh activation ie. the final output of the encoder for the first word.
    hidden_reps, cls_head = bert_model(token_ids, attn_mask, seg_ids)
    print(hidden_reps)
    print(cls_head)


execute()


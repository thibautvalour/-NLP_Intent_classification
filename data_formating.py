import numpy as np
from collections import defaultdict
from transformers import BertTokenizer
import datasets
from torch.utils.data import Dataset, DataLoader


def tokenize_pad_numericalize_dialog(entry, tokenizer, vocab_stoi, max_length):
  ''' message level '''
  dialog = [[vocab_stoi['[CLS]']]
            + [vocab_stoi[token] if token in vocab_stoi else vocab_stoi['[UNK]'] 
               for token in tokenizer.tokenize(e.lower()) ] 
            + [vocab_stoi['[SEP]']]
            for e in entry]
  padded_dialog = list()
  for d in dialog:
    if len(d) < max_length: 
        padded_dialog.append(d + [ vocab_stoi['[PAD]'] 
                                  for _ in range(len(d), max_length)])
    elif len(d) > max_length:
        padded_dialog.append(d[:max_length])
    else:           
         padded_dialog.append(d) 
  
  return padded_dialog

def tokenize_all_dialog(entries, tokenizer, vocab_stoi, max_message_length,
                        max_dialog_length):
    ''' dialog level '''
    pad_message = [ vocab_stoi['[PAD]'] ]
    pad_label = [4] # because our labels go from 0 to 3
    res_dialog, res_labels = [], []

    # Group messages by dialogue ID
    dialogues = defaultdict(list)
    dialogue_labels = defaultdict(list)
    for i in range(len(entries['text'])):
        dialogue_id = entries['Dialogue_ID'][i]
        dialogues[dialogue_id].append(entries['text'][i])
        dialogue_labels[dialogue_id].append(entries['labels'][i])

    # Tokenize and pad messages for each dialogue
    for dialogue_id, text in dialogues.items():
        labels = dialogue_labels[dialogue_id]
        text = tokenize_pad_numericalize_dialog(text, tokenizer, vocab_stoi,
                                                max_length=max_message_length)
        if len(text) < max_dialog_length:
            text = text + [[vocab_stoi['[PAD]']] * max_message_length 
                           for i in range(len(text), max_dialog_length)]
            labels = labels + pad_label * (max_dialog_length - len(labels))
        elif len(text) > max_dialog_length:
            text = text[-max_dialog_length:]
            labels = labels[-max_dialog_length:]
        res_dialog.append(text)
        res_labels.append(labels)

    res = {'text': res_dialog, 'labels': res_labels}
    return res

class DialogActDataset(Dataset):
    def __init__(self, data, args):
      self.args = args
      self.data = data

    def __len__(self):
      return len(self.data)
    
    def __getitem__(self, idx):
      item = {
          "text": np.array(self.data[idx]['text']),
          "label": np.array(self.data[idx]['labels'])
      }
      return item
    
def make_dataloader(dataset_train, dataset_test, args):
    dataset = datasets.DatasetDict({"train" : dataset_train, "test": dataset_test})
    dataset = dataset.rename_column("Label", "labels")
    dataset = dataset.rename_column("Utterance", "text")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    vocab_stoi = tokenizer.get_vocab()
    encoded_dataset = dataset.map(lambda e: tokenize_all_dialog(e, tokenizer, vocab_stoi,
                                                                args['max_word'], args['max_sentence']),
                                  batched=True, remove_columns=['Dialogue_Act','Dialogue_ID','Idx'])
    encoded_dataset.set_format("torch")

    train_loader = DataLoader(DialogActDataset(encoded_dataset['train'], args),
                               batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_loader  = DataLoader(DialogActDataset(encoded_dataset['test'], args),
                               batch_size=args['batch_size'], shuffle=True, drop_last=True)
    
    return train_loader, test_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertBiLSTM(nn.Module):

    def __init__(self, args):
        super(BertBiLSTM, self).__init__()

        self.args = args

        # Load Bert Model
        self.bert_model = BertModel.from_pretrained('bert-base-cased')
        # Freeze some Bert Layers
        for param in self.bert_model.parameters():
            param.requires_grad = False
        # Unfreeze the last layer
        for param in self.bert_model.encoder.layer[-args['unfreezed_bert_layer']:].parameters():
            param.requires_grad = True

        # Dimensions are divided by 2 due to bidirectional being True
        self.lstm = nn.LSTM(input_size=args['bert_output_size'],
                            hidden_size=args['lstm_hidden_dimension']//2, 
                            num_layers=args['lstm_layers'],
                            batch_first=True,
                            bidirectional=True)
        
        self.dropout = nn.Dropout(p=args['dropout'])
        self.hidden2intent = nn.Linear(args['lstm_hidden_dimension'], args['num_class'])

    def forward(self, batch):
        input_ids = batch['text'].to(self.args['device'])
        labels = batch['label'].to(self.args['device'])

        input_ids = input_ids.view(self.args['bsize']*self.args['max_sentence'], self.args['max_word'])
        outputs = self.bert_model(input_ids)['pooler_output']
        outputs = outputs.view(self.args['bsize'], self.args['max_sentence'], -1)
 
        outputs = self.dropout(outputs)
        outputs, _ = self.lstm(outputs)
        outputs = self.dropout(outputs)
        outputs = self.hidden2intent(outputs)
        outputs = outputs.view(-1, self.args['num_class'])
        logits = F.log_softmax(outputs, 1)

        loss_fn = nn.NLLLoss(weight=self.args['class_weights'])  
        loss = loss_fn(logits, labels.view(-1)) 

        # Compute accuracy without 4 labels (PAD)
        _, tag_seq  = torch.max(logits, 1)
        labels = labels.view(-1)
        tag_seq = tag_seq.view(-1)

        mask = labels != 4
        labels, tag_seq = labels[mask], tag_seq[mask]
        accuracy = (labels == tag_seq).sum().item() / len(labels)

        return loss, logits, tag_seq, accuracy

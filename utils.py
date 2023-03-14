import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, optimizer, epoch, train_loader):

  model.train()
  training_dict = {'loss': [], 'acc': []}

  for batch in tqdm(train_loader, desc="Training, Epoch %s:" % (epoch),
                    position=0 ,leave=True):
    optimizer.zero_grad()
    model.zero_grad()
    loss, logits, tag_seq, accuracy = model(batch)
    loss.backward()
    optimizer.step()
    training_dict['loss'].append(loss.item())
    training_dict['acc'].append(accuracy)
  
  avg_loss = sum(training_dict['loss'])/len(training_dict['loss'])
  avg_acc = sum(training_dict['acc'])/len(training_dict['acc'])
  return avg_loss, avg_acc

@torch.no_grad()
def test(model, epoch, test_loader):
  model.eval()
  test_dict = {'loss': [], 'acc': []}

  for batch in tqdm(test_loader, desc="Test, Epoch %s:" % (epoch), 
                    position=0 ,leave=True):
    loss, logits, tag_seq, accuracy = model(batch)
    test_dict['loss'].append(loss.item())
    test_dict['acc'].append(accuracy)
  
  avg_loss = sum(test_dict['loss'])/len(test_dict['loss'])
  avg_acc = sum(test_dict['acc'])/len(test_dict['acc'])

  return avg_loss, avg_acc

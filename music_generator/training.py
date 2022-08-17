import sys
from time import time

import numpy as np

import torch
import torch.nn as nn




def transformBatch(batch):
    length = batch.shape[1]
    src = batch[:, :length-1]
    trg = batch[:, 1:]
    return src.transpose(0, 1), trg.transpose(0, 1)




def trainTransformer(model, train_data, optimiser, criterion, device, debug=False, MAX_LENGTH=32, update=50):
    model.train()
    
    total_loss = 0
    start_time = time()
    
    src_msk = model.generate_square_subsequent_mask(MAX_LENGTH - 1)

    losses = []
    batch_losses = []
    
    for i, batch in enumerate(train_data.dataset):
        src, trg = transformBatch(batch)
        optimiser.zero_grad()
        
        #src_key_padding_mask = (src == PADDING).transpose(0, 1)

        output = model(src, src_msk)#, src_key_padding_mask)
        '''
        output                 -> (src_seq_length, batch, vocab_size)
        output.view(-1, VOCAB) -> (src_deq_length * batch_size, vocab_size)
        trg.flatten()          -> (src_deq_length * batch_size)
        '''
        loss = criterion(output.flatten(), trg.flatten())  

        if not debug:
            loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimiser.step()
        
        losses.append(loss.item())
        
        if i % update == 0 and i > 0:
            curr_loss = np.mean(losses)
            elapsed = time() - start_time
            sys.stdout.write('    {:3d}/{:3d} | time {:5.2f} | mean batch loss {:5.2f}\r'.format(
                i, len(train_data.dataset), elapsed, curr_loss
            ))
            batch_losses.append(curr_loss)
            start_time = time()
            losses = []
            
        if debug:
            break
    
    return np.mean(batch_losses), batch_losses

    
def evaluateTransformer(model, test_data, criterion, MAX_LENGTH=32):
    model.eval()
    losses = []
    
    src_msk = model.generate_square_subsequent_mask(MAX_LENGTH - 1)
    
    with torch.no_grad():
        for batch in test_data.dataset:
            src, trg = transformBatch(batch)
            
            output = model(src, src_msk)
            losses.append(criterion(output.flatten(), trg.flatten()).item())
            
    return np.mean(losses)

import os
import warnings
import pickle as pkl
from time import time
from importlib import reload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import librosa

import torch
import torchaudio


SAMPLES = './LAWKI_NOW_samples/'
device = torch.device('cpu')
print('computing on', device)


# collect filepaths of audio samples

fps = []
for root, _, file in os.walk(SAMPLES):
    for fn in file:
        if '.wav' in fn:
            fps.append(os.path.abspath(os.path.join(root, fn)))
            
print(f'Found {len(fps)} samples in {os.path.abspath(SAMPLES)}')

from feature_extractor import feature_extractor_from_file

unit_data = []

for fp in tqdm(fps):
    data = feature_extractor_from_file(fp)
    unit_data.append(data)

feature_tensors = np.stack([data['features'] for data in unit_data])
feature_tensors = torch.tensor(feature_tensors, dtype=torch.float, device=device)

unit_df = pd.DataFrame(unit_data).drop('features', 1)

print('MAKING MODEL')

from models import LinearAE

embedding_params = {
    'feature_size': feature_tensors.shape[1],
    'hidden_size': 32,
    'latent_size': 16,
    'dropout': 0.4
}

embedding_model = LinearAE(**embedding_params).to(device)

BATCH_SIZE = 16
EPOCHS = 500


lr = 0.0001

optimiser = torch.optim.Adam(embedding_model.parameters(), lr=lr)
cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
criterion = torch.nn.SmoothL1Loss(reduction='mean')


train_num = int(len(feature_tensors) * 0.7)
val_num = len(feature_tensors) - train_num

train_subset, val_subset = torch.utils.data.random_split(feature_tensors, [train_num, val_num])

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)

def step(model, data, training=True):
    Q = data.to(device)
    if training:
        optimiser.zero_grad()
        
    R, _ = model(Q) 
    loss = criterion(R, Q)
    
    return loss

def fit(model, dataset):
    model.train()
    losses = []
    
    for i, data in enumerate(dataset):
        loss = step(model, data)
        losses.append(loss.item())
        loss.backward()
        optimiser.step()
        
        if i % 50 == 0 and i > 0:
            print('\t batch {:4d} | loss {:.4f}'.format(i, np.mean(losses[-50:])))
            
    return np.mean(losses[-50:])

def validate(model, dataset):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for i, data in enumerate(dataset):
            loss = step(model, data, training=False)
            losses.append(loss.item())
            
    return np.mean(losses)


train_losses = []
val_losses = []


def train(model):
    train_start = time()
    print(' == Started Training model ==')

    for epoch in range(EPOCHS):
        print('Epoch {:2} of {}:'.format(epoch+1, EPOCHS))
        start = time()

        train_ds_losses = []
        val_ds_losses = []
        
        train_loss = fit(model, train_loader)
        val_loss = validate(model, val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        end_time = time() - start
        total_time = time() - train_start
        total_text = '{:02d}:{:02d}'.format(int(total_time//60), int(total_time%60))

        print('\t\t train: {:.4f} | validation: {:.4f} | epoch in {:3.2f}s | total time {}'.format(
            train_loss, val_loss, end_time, total_text))



try:
    train(embedding_model)
except KeyboardInterrupt:
    print('Training paused')


save_path = f'./models/{embedding_model.model_type}_embedding_model.pt'

print('Saving model to', save_path)
if not os.path.exists(os.path.split(save_path)[0]):
    print('    making dir', os.path.split(save_path)[0])
    os.mkdir(os.path.split(save_path)[0], )

torch.save({
    'model_state_dict': embedding_model.state_dict(),
    'model_params': embedding_params,
    'optimiser_state_dict': optimiser.state_dict(),
    'training_loss': train_losses,
    'validation_loss': val_losses,
}, save_path)



with torch.no_grad():
    embedded = embedding_model.encode(feature_tensors.to(device))


print('FITTING UMAP')

import umap

reducer = umap.UMAP(verbose=True, n_neighbors=100, min_dist=0.99, n_components=2)

reducer.fit(embedded.cpu().numpy())

reduced = reducer.transform(embedded.cpu().numpy())

import json
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(reduced)
scaled = scaler.transform(reduced)

points = [pos.tolist() for pos in scaled]

data = {'points': points, 'fps': fps}

with open('./audio_vis/data.json', 'w') as f:
    f.write('d = ' + json.dumps(data))


with open('./models/reducer.pkl', 'wb') as f:
    pkl.dump(reducer, f)

    with open('./models/reducer_scaler.pkl', 'wb') as f:
        pkl.dump(scaler, f)


print('DONE')



















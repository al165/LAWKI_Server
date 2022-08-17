import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearAE(nn.Module):
    def __init__(self, feature_size, hidden_size, latent_size, dropout=0.2):
        super(LinearAE, self).__init__()

        self.model_type = 'AE'

        self._latent_size = latent_size

        # Encoder:
        self.enc1 = nn.Linear(in_features=feature_size, out_features=hidden_size)
        self.enc2 = nn.Linear(in_features=hidden_size, out_features=latent_size)

        # Decoder:
        self.dec1 = nn.Linear(in_features=latent_size, out_features=hidden_size)
        self.dec2 = nn.Linear(in_features=hidden_size, out_features=feature_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        z = self.encode(x)
        R = self.decode(z)

        return R, z

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = self.dropout(x)
        z = self.enc2(x) # .view(-1, self._latent_size)
        return z

    def decode(self, z):
        x = F.relu(self.dec1(z))
        x = self.dropout(x)
        R = self.dec2(x) #.unsqueeze(1)
        return R


class LinearVAE(nn.Module):
    def __init__(self, feature_size, hidden_size, latent_size, dropout=0.2):
        super(LinearVAE, self).__init__()

        self.model_type = 'VAE'

        self._latent_size = latent_size

        # Encoders:
        sizes = [feature_size]
        if type(hidden_size) == int:
            sizes.append(hidden_size)
        else:
            try:
                sizes.extend(hidden_size)
            except TypeError:
                raise TypeError('`hidden_size` must be int or iterable')

        self.encoders = nn.ModuleList()
        for i, in_f in enumerate(sizes[:-1]):
            out_f = sizes[i+1]
            self.encoders.append(nn.Linear(in_features=in_f, out_features=out_f))

        self.enc_mu = nn.Linear(in_features=sizes[-1], out_features=latent_size)
        self.enc_log_var = nn.Linear(in_features=sizes[-1], out_features=latent_size)

        # Decoder:
        self.decoders = nn.ModuleList()
        sizes.append(latent_size)
        sizes = list(reversed(sizes))
        for i, in_f in enumerate(sizes[:-1]):
            out_f = sizes[i+1]
            self.decoders.append(nn.Linear(in_features=in_f, out_features=out_f))

        #self.dec1 = nn.Linear(in_features=latent_size, out_features=hidden_size)
        #self.dec2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        #self.dec3 = nn.Linear(in_features=hidden_size, out_features=feature_size)

        self.dropout = nn.Dropout(p=dropout)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        #x = F.relu(self.enc1(x))
        #x = self.dropout(x)
        #x = F.relu(self.enc2(x))
        #x = self.dropout(x)

        for enc in self.encoders:
            x = F.relu(enc(x))
            x = self.dropout(x)

        mu = self.enc_mu(x)
        log_var = self.enc_log_var(x)
        sigma = torch.exp(log_var)

        z = mu + sigma * self.N.sample(mu.shape)

        self.kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - sigma, dim=1), dim=0)

        y = mu
        for dec in self.decoders[:-1]:
            y = F.relu(dec(y))
            y = self.dropout(y)

        y = self.decoders[-1](y)
        #y = F.relu(self.dec1(mu))
        #y = self.dropout(y)
        #y = F.relu(self.dec2(y))
        #y = self.dropout(y)
        #y = self.dec3(y)

        return y, z

    def encode(self, x):
        '''Deterministic encoding into latent space.'''
        #x = F.relu(self.enc1(x))
        #x = F.relu(self.enc2(x))
        #z = self.enc3(x)

        for enc in self.encoders:
            x = F.relu(enc(x))
            x = self.dropout(x)

        mu = self.enc_mu(x)

        return mu

    def decode(self, z):
        #z = F.relu(self.dec1(z))
        #z = F.relu(self.dec2(z))
        #y = self.dec3(z)

        y = z
        for dec in self.decoders[:-1]:
            y = F.relu(dec(y))
            y = self.dropout(y)

        y = self.decoders[-1](y)

        return y


class SimpleSemanticRelevance(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(SimpleSemanticRelevance, self).__init__()

        self._layer1 = nn.Linear(input_size*2, hidden_size)
        self._layer2 = nn.Linear(hidden_size, 1)

        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)


    def forward(self, x, context):
        '''
        param x: (batch_size, embedding_size)
        param context: (batch_size, embedding_size)
        '''

        x = self._layer1(torch.cat([x, context], dim=1))
        x = self._relu(x)
        x = self._dropout(x)
        x = self._layer2(x)
        # nn.sigmoid is computed by the criterion BCEWithLogitsLoss

        return x


class TransformerModel(nn.Module):

    def __init__(self, ntokens, ninp, nhead, nhid, nlayers, padding_idx=None, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = 'Transformer'

        self.encoder = nn.Embedding(ntokens, ninp, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(ninp, dropout, 512)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntokens)

        self.init_weights()


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        '''
        param: src                  (src_seq_length, batch_size)
        param: src_mask             (src_seq_length, src_seq_length)
        param: src_key_padding_mask (batch_size, src_seq_length)

        returns: output             (batch_size, )
        '''

        # src -> (src_seq_length, batch, embed_size):
        src = src[-512:, :]
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        # output -> (src_seq_length, batch, embed_size)
        output = self.transformer_encoder(src, src_mask,
                                          src_key_padding_mask=src_key_padding_mask)

        # output -> (src_seq_length, batch, vocab_size)
        output = self.decoder(output)

        return output



class EmbeddedTransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, padding_idx=None, dropout=0.5):
        '''A Transformer model without the embedding and decoding layers'''

        super(EmbeddedTransformerModel, self).__init__()

        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout, 512)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)


    def generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)


    def forward(self, src, src_mask=None, src_key_padding_mask=None, max_length=512):
        '''
        param: src                  (src_seq_length, batch_size, embedding_size)
        param: src_mask             (src_seq_length, src_seq_length)
        param: src_key_padding_mask (batch_size, src_seq_length)
        param: max_length           int, default 512

        returns: output             (batch_size,)
        '''

        # src -> (src_seq_length, batch, embed_size):
        src = src[-max_length:, :]
        src = self.pos_encoder(src)

        # output -> (src_seq_length, batch, embed_size)
        output = self.transformer_encoder(src, src_mask,
                                          src_key_padding_mask=src_key_padding_mask)


        return output



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

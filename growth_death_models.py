import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Decoder(nn.Module):
    def __init__(self, input_dim, seq_length, device=None):
        super(Decoder, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.device = device

        self.linear1 = nn.Linear(input_dim*seq_length, input_dim*5)
        self.norm1 = nn.LayerNorm(input_dim*5)
        self.gelu1 = nn.GELU()
        self.linear2 = nn.Linear(input_dim*5, 10)
        self.norm2 = nn.LayerNorm(10)
        self.gelu2 = nn.GELU()
        self.out = nn.Linear(10, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        decoder_output = x.reshape(batch_size, self.S * self.D)
        decoder_output = self.linear1(decoder_output)
        decoder_output = self.norm1(decoder_output)
        decoder_output = self.gelu1(decoder_output)
        decoder_output = self.linear2(decoder_output)
        decoder_output = self.norm2(decoder_output)
        decoder_output = self.gelu2(decoder_output)
        output = self.out(decoder_output)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # feature num

        self.W_Q = nn.Linear(self.D, self.D * self.H)
        self.W_K = nn.Linear(self.D, self.D * self.H)
        self.W_V = nn.Linear(self.D, self.D * self.H)
        self.W_O = nn.Linear(self.D * self.H, self.D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))  # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)  # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x):
        Q = self.W_Q(x)  # (B, S, D)
        K = self.W_K(x)  # (B, S, D)
        V = self.W_V(x)  # (B, S, D)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_weights, V)  # (B, H, S, D)
        output = self.concat_heads(output)  # (B, S, D*H)
        output = self.W_O(output)
        return output

class Transformer_Encoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Transformer_Encoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        attention_output = self.attention(x)
        output = x + attention_output
        output = self.linear(output)
        output = self.norm(output)
        output = self.gelu(output)
        output = self.out(output)
        return output
#

class Transformer(nn.Module):
    def __init__(self, input_dim, seq_length, device=None):
        super(Transformer, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.device = device
        self.heads_num = 16

        self.encoder0 = Transformer_Encoder(input_dim, self.heads_num)
        self.encoder1 = Transformer_Encoder(input_dim, self.heads_num)
        self.encoder2 = Transformer_Encoder(input_dim, self.heads_num)
        self.decoder = Decoder(input_dim, seq_length, device)

    def forward(self, x):
        encoder_output = self.encoder0(x)
        encoder_output = self.encoder1(encoder_output)
        encoder_output = self.encoder2(encoder_output)
        encoder_output = x + encoder_output
        batch_length = encoder_output.shape[0]
        encoder_output = encoder_output.reshape(batch_length, self.S * self.D)
        outputs = self.decoder(encoder_output)
        return outputs

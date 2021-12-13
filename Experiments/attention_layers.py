
import torch
from torch import nn

import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        self.act = nn.LeakyReLU(inplace=False)
    def forward(self, s, enc_output):
        enc_output = enc_output.unsqueeze(0) 
        batch_size = enc_output.shape[1] 
        src_len = enc_output.shape[1]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        s = s.view(s.shape[0],src_len,enc_output.shape[2])
        score = self.act(self.attn(torch.cat((s, enc_output), dim=2)))
        attention = self.v(score).squeeze(2)
        return F.softmax(attention, dim=1)




import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from layers import MLP
from utils import one_hot_encode


class MSKT(nn.Module):

    def __init__(self, concept_num, cognitive_num, question_num, hidden_dim, disc,
                 dropout=0.5, bias=True, has_cuda=False, device='cpu'):
        super(MSKT, self).__init__()
        self.concept_num = concept_num
        self.cognitive_num = cognitive_num
        self.question_num = question_num
        self.hidden_dim = hidden_dim

        self.disc = disc

        self.dropout = dropout
        self.bias = bias 

        self.has_cuda = has_cuda
        self.device = device

        self.knowledge_emb = nn.Embedding(self.concept_num+1, self.hidden_dim)
        # self.paper_emb = nn.Embedding(2+1, self.hidden_dim)
        self.question_emb = nn.Embedding(self.question_num+1, self.hidden_dim)
        self.res_emb = nn.Embedding(2+1, self.hidden_dim)
        self.cog_emb = nn.Embedding(self.cognitive_num+1, self.hidden_dim)

        self.fc0 = nn.Linear(self.hidden_dim*4, self.hidden_dim, bias=self.bias)
        self.lstm = nn.LSTM(self.hidden_dim*2, self.hidden_dim, bias=self.bias)
        self.fc1 = nn.Linear(self.hidden_dim, self.concept_num, bias=self.bias)


    def _get_next_pred(self, features, difficulties, yt):
        r"""
        Parameters:
            features: the sequences of concept
            difficulties: the sequences of difficulty levels
            yt: predicted knowledge state

        Shape:
            features: [batch_size, seq_len-1]
            difficulties: [batch_size, seq_len-1]
            yt: [batch_size, seq_len-1, concept_num]

        Return:
            pred: predicted correct probability of each question answered at the next timestamp
        """

        index =  F.one_hot(features, num_classes=self.concept_num+1)    # [batch_size, seq_len-1, concept_num+1]
        
        res = torch.sum(yt * index[:,:,:-1], dim=-1)    # [batch_size, seq_len-1]
        index = torch.sum(index[:,:,:-1], dim=-1)       # [batch_size, seq_len-1]

        index[index<=0.0] = 1.0
        res = res / index                               # [batch_size, seq_len-1]
        res = torch.sigmoid(self.disc * (res - (difficulties.float()+0.5) / self.cognitive_num))
        # [batch_size, seq_len-1]

        return res


    def forward(self, features, questions, difficulties, papers, answers):
        r"""
        Parameters:
            features: 
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        
        Shape:
            features/questions/difficulties/papers/answers: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]

        Return:
            pred_res: the correct probability of questions answered at the next timestamp
        """
        batch_size, seq_len = features.shape
        
        features[features<0] = self.concept_num
        questions[questions<0] = self.question_num
        difficulties[difficulties<0] = self.cognitive_num
        papers[papers<0] = 2
        answers[answers<0] = 2
        
        ft = self.knowledge_emb(features.long())
        qt = self.question_emb(questions.long())
        dt = self.cog_emb(difficulties.long())
        pt = torch.clamp(papers, max=1).long()
        at = self.res_emb(answers.long())
        # [batch_size, seq_len, hidden_dim]

        
        inputs = torch.cat((ft, qt, dt, at), dim=-1) 
        # [batch_size, seq_len, hidden_dim * 4]
        inputs = F.relu(self.fc0(inputs))    # [batch_size, seq_len, hidden_dim]
        
        inputs = torch.cat((inputs,inputs), dim=-1)
        inputs = one_hot_encode(pt, self.hidden_dim) * inputs    # [batch_size, seq_len, hidden_dim*2]
        
        outputs, (_, _) = self.lstm(inputs, None)    # [batch_size, seq_len, hidden_dim]
        yt = torch.sigmoid(self.fc1(outputs))    # [batch_size, seq_len, concept_num]


        if seq_len > 1:
            pred_res =  self._get_next_pred(features[:, 1:], difficulties[:, 1:], yt[:, :-1, :]) 
            # [batch_size, seq_len - 1]
        else:
            pred_res = None
        
        # print(pred_res.shape)
        # print(pred_res)
        # has_nan = torch.isnan(pred_res).any()
        # print(has_nan)
        return outputs, yt, pred_res

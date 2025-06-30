import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils import accuracy


class KTLoss(nn.Module):

    def __init__(self, exam_dic, k1=0.0, k2=0.0):
        super(KTLoss, self).__init__()
        self.k1 = k1    
        self.k2 = k2    
        self.exam_dic = exam_dic    # the ideal score

    def forward(self, students, pred_answers, real_answers, ht, ht_trans, yt, need_score):
        r"""
        Parameters:
            students: student_id 
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
            ht/ht_trans: hidden state
            yt: knowledge state
            need_score: need diagnosis
        Shape:
            students: [batch_size, seq_len]
            pred_answers: [batch_size, seq_len - 1]
            real_answers: [batch_size, seq_len]
            ht/ht_trans: [batch_size, seq_len, hidden_dim]
            yt: [batch_size, seq_len, concept_num]
            need_score: [batch_size, seq_len]
        Return:
        """

        
        batch_size = real_answers.shape[0]
        state_mask = torch.ne(real_answers, -1) & torch.ne(real_answers, 2)    # [batch_size, seq_len] 
        real_answers = real_answers[:, 1:]  # timestamp=1 ~ T
        # real_answers shape: [batch_size, seq_len - 1]
        # Here we can directly use nn.BCELoss, but this loss doesn't have ignore_index function
        answer_mask = torch.ne(real_answers, -1) & torch.ne(real_answers, 2)   # [batch_size, seq_len - 1] 
        pred_one, pred_zero = pred_answers, 1.0 - pred_answers  # [batch_size, seq_len - 1]

        # calculate auc and accuracy metrics
        try:
            y_true = real_answers[answer_mask].cpu().detach().numpy()
            y_pred = pred_one[answer_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)  # may raise ValueError
            output = torch.cat((pred_zero[answer_mask].reshape(-1, 1), pred_one[answer_mask].reshape(-1, 1)), dim=1)
            label = real_answers[answer_mask].reshape(-1, 1)
            acc = accuracy(output, label)
            acc = float(acc.cpu().detach().numpy())
        except ValueError as e:
            auc, acc = -1, -1

        loss_func = nn.BCELoss()  # ignore masked values in real_answers
        loss = loss_func(pred_answers[answer_mask], real_answers[answer_mask])


        need_score[(need_score == -1) | (need_score == 2)] = 0
        judge = torch.sum(need_score, dim=-1)    # [batch_size] calculate the number of the exams done by each student

        for b in range(batch_size):
            student_id = students[b][0].item()
            y = yt[b][need_score[b]==1]    # [paper_num, concept_num]
            exam_res = self.exam_dic[student_id]    # [paper_num, concept_num]
            exam_res = torch.tensor(exam_res, device=y.device)
            origin_ht = ht[b][need_score[b]==1]    # [paper_num, hidden_dim]
            trans_ht = ht_trans[b][need_score[b]==1]    # [paper_num, hidden_dim]
            assert judge[b].item() == exam_res.shape[0]
            assert judge[b].item() == y.shape[0]
            assert judge[b].item() == origin_ht.shape[0]
            assert origin_ht.shape == trans_ht.shape

            loss += judge_score(y, exam_res) * self.k1 / batch_size
            loss += ContrastiveLoss(origin_ht, trans_ht) * self.k2 / batch_size
        
        
        return loss, auc, acc


def ContrastiveLoss(origin, trans):
    r"""
    Parameters:
        origin: the origin sequence
        trans: the shuffled sequence

    Shape:
        origin: [paper_num, hidden_dim]
        trans: [paper_num, hidden_dim]

    Return:
        loss
    """
    
    # negative pairs
    neg_sim = F.cosine_similarity(origin.unsqueeze(1), origin.unsqueeze(0), dim=2)
    # positive pairs
    pos_sim = F.cosine_similarity(origin, trans, dim=-1)    # [paper_num]

    identity_matrix = torch.eye(neg_sim.shape[0], dtype=neg_sim.dtype, device=neg_sim.device)
    neg = torch.sum(torch.exp(neg_sim) * (1.0-identity_matrix), dim=-1)    # [paper_num]
    pos = torch.exp(pos_sim)    # [paper_num]
    loss = torch.sum(pos / (pos + neg), dim=-1)    
    loss = -torch.log(loss)

    return loss



def judge_score(y, exam_res):
    r"""
    Parameters:
        y: the predicted knowledge state
        exam_res: the ideal score of each exam

    Shape:
        y: [paper_num, concept_num]
        exam_res: [paper_num, concept_num]

    Return:
        loss
    """

    loss_func = nn.MSELoss()
    loss = loss_func(y[exam_res != -1.0], exam_res[exam_res != -1.0])
    return loss
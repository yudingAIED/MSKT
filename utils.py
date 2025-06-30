import itertools
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp

# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)

# get the ideal scores within one exam
def real_score(exam_dict, concept_num):
    
    """
    features [batch_size, seq_len]
    concept_num: the number of all concepts
    """
    
    score = [-1.0] * concept_num    # [concept_num]
    for key in exam_dict.keys():
        score[key] = exam_dict[key]    
    return score

# shuffle the learning records within one exam
def paper_shuffle(questions, features, difficulties, papers, answers):

    """
    questions [batch_size, seq_len]
    """
    batch_size, seq_len = papers.shape
    # q_mask = torch.ne(questions, -1)    # [batch_size, seq_len]

    new_questions = questions.clone()
    new_features = features.clone()
    new_difficulties = difficulties.clone()
    new_papers = papers.clone()
    new_answers = answers.clone()

    
    for k in range(batch_size):
        paper = papers[k]
        condition = (paper == 1) | (paper == -1)
        indices = condition.nonzero(as_tuple=True)[0].tolist()
        for n in range(0, len(indices)-1):
            row_indices = (torch.randperm(indices[n+1]-indices[n]) + indices[n]).to(paper.device)
            
            new_questions[k][indices[n]:indices[n+1]] = questions[k][row_indices]
            new_features[k][indices[n]:indices[n+1]] = features[k][row_indices]
            new_difficulties[k][indices[n]:indices[n+1]] = difficulties[k][row_indices]
            new_answers[k][indices[n]:indices[n+1]] = answers[k][row_indices]

    return new_questions, new_features, new_difficulties, new_papers, new_answers

# encode the embedding according to whether the record belongs to a new exam
def one_hot_encode(papers, hidden_dim):
    """
    papers [batch_size, seq_len]
    """

    rule_0 = torch.tensor([1] * hidden_dim + [0] * hidden_dim).float().to(papers.device)
    rule_1 = torch.tensor([0] * hidden_dim + [1] * hidden_dim).float().to(papers.device)
    result = (rule_0 * (papers == 0).float().unsqueeze(-1)) + (rule_1 * (papers == 1).float().unsqueeze(-1))
    return result


# check whether the record belongs a new exam (0-False 1-True)
def is_new_paper(papers):
    
    """
    papers [batch_size, seq_len]

    """
    batch_size, seq_len = papers.shape
    
    new_papers = torch.ones((batch_size, seq_len), device=papers.device).long()
    need_score = torch.ones((batch_size, seq_len), device=papers.device).long()
    
    new_papers[:,1:][papers[:,:-1] == papers[:,1:]] = 0
    new_papers[papers==-1] = -1
    need_score[:,:-1][papers[:,:-1] == papers[:,1:]] = 0
    need_score[papers==-1] = -1

    return new_papers, need_score

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from utils import real_score

class KTDataset(Dataset):
    def __init__(self, students, questions, features, difficulties, papers, answers):
        super(KTDataset, self).__init__()
        self.students = students           # student ID
        self.questions = questions         # question ID
        self.features = features           # concept ID
        self.difficulties = difficulties   # difficulty level
        self.papers = papers               # exam ID
        self.answers = answers             # answer result

    def __getitem__(self, index):
        return self.students[index], self.questions[index], self.features[index], self.difficulties[index], \
            self.papers[index], self.answers[index]

    def __len__(self):
        return len(self.questions)


def pad_collate(batch):
    (students, questions, features, difficulties, papers, answers) = zip(*batch)
    students = [torch.LongTensor(stu) for stu in students]
    questions = [torch.LongTensor(que) for que in questions]
    features = [torch.LongTensor(feat) for feat in features]
    difficulties = [torch.FloatTensor(diff) for diff in difficulties]
    papers = [torch.LongTensor(paper) for paper in papers]
    answers = [torch.FloatTensor(ans) for ans in answers]

    student_pad = pad_sequence(students, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    difficulty_pad = pad_sequence(difficulties, batch_first=True, padding_value=-1)
    paper_pad = pad_sequence(papers, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return student_pad, question_pad, feature_pad, difficulty_pad, paper_pad, answer_pad


def my_load_dataset(file_path, dataset_file_path, batch_size, train_ratio=0.7, val_ratio=0.2,
                    shuffle=True, use_cuda=True, device='cpu'):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        exercise_file_path: input file path of exercise data
        batch_size: the size of a student batch
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    """


    dataset_path = os.path.join(file_path, dataset_file_path)
    df = pd.read_csv(dataset_path)
    record_num = df.shape[0]
    print('record_num: ', record_num)

    df['correct'] = df['correct'].astype(int)
    df['difficulty'] = df['difficulty'].astype(int)
    

    # 分不同的数据源进行处理
    if dataset_file_path == "new_college_physics.csv":
        concept_num = 360
    elif dataset_file_path == "new_math.csv":
        concept_num = 39
    elif dataset_file_path == "new_phy.csv":
        concept_num = 53
    

    # Data Process
    df.drop(df[df['concept_id'] == '#N/A'].index)


    df = df.groupby('student_id').filter(lambda q: len(q) > 1).copy()    # 只取长度大于1（这样用KT才有意义）


    df['concept_id'] = df['concept_id'].astype(int)
    df['question_id'] = df['question_id'].astype(int)
    df['paper_id'] = df['paper_id'].astype(int)
    


    student_list = []
    question_list = []
    feature_list = []
    difficulty_list = []
    answer_list = []
    paper_list = []
    seq_len_list = []

    def get_data(series):
        student_list.append(series['student_id'].tolist())
        question_list.append(series['question_id'].tolist())
        feature_list.append(series['concept_id'].tolist())
        difficulty_list.append(series['difficulty'].tolist())
        answer_list.append(series['correct'].astype('float').tolist())
        paper_list.append(series['paper_id'].tolist())
        seq_len_list.append(series['correct'].shape[0])



    df.groupby('student_id').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    mean_seq_len = np.mean(seq_len_list)
    print('max_seq_len: ', max_seq_len)
    print('mean_seq_len: ', mean_seq_len)
    
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    

    exercise_num = df['question_id'].max() + 1
    print('exercise_num: ', exercise_num)
    print('concept_num: ', concept_num)

    data_size = len(seq_len_list)

    
    kt_dataset = KTDataset(student_list, question_list, feature_list, difficulty_list, paper_list, answer_list)
    train_size = int(train_ratio * data_size)
    val_size = data_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(kt_dataset,
                                                                             [train_size, val_size])
    print('train_size: ', train_size, 'val_size: ', val_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    return student_num, concept_num, exercise_num, train_data_loader, valid_data_loader, test_data_loader

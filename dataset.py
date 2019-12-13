import numpy as np
import pandas as pd
import random

import re

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer

class BertDataset(Dataset):
    
    def __init__(self, train_dir, test_dir, doc_dir, qrydoc_pair, pos=None, split=None):
        """
        :param train_dir: training data directory path
        :param test_dir: tresting data directory path
        :param doc_dir: document data directory path
        :param qrydoc_pair: dictionary for every query to every document
        :param pos: positive documents for Discriminator
        :param split: Whcih type the dataset is ex)Gen, Dis, Test
        """

        # Setting directory path
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.doc_dir = doc_dir

        # Setting query-document pair of dataset and positive documents
        self.qrydoc_pair = qrydoc_pair
        self.query_list = list(qrydoc_pair.keys())
        self.doc_size = len(qrydoc_pair[self.query_list[0]])
        self.pos = pos

        # Setting tokenizer of BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Setting flag for Generator, Discriminator, Test dataset
        self.split = split
    
    def _preprocess(self, query, doc):
        # Query   
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        #query = re.sub(r'[^\w]', ' ', query)
        #query = query.split()
        query = f"{cls_token} {query} {sep_token}"
        query_tokens = self.tokenizer.tokenize(query)
        query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
        cls_ids = self.tokenizer.convert_tokens_to_ids(cls_token)
        sep_ids = self.tokenizer.convert_tokens_to_ids(sep_token)

        query_ids = [cls_ids] + query_ids + [sep_ids]

        # Document
        doc_len = 512 - len(query_ids)
        #doc = re.sub(r'[^\w]', ' ', doc)
        #doc = doc.split()
        #doc = f"{' '.join(doc)}"
        doc_tokens = self.tokenizer.tokenize(doc)
        doc_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
        doc_ids = doc_ids[:doc_len]
        
        ids = query_ids + doc_ids      
        
        token_type = [*(0 for _ in query_ids), *(1 for _ in doc_ids)]
        assert len(token_type) == len(ids)

        attention_mask = [*(1 for _ in query_ids), *(1 for _ in doc_ids)]
        assert len(attention_mask) == len(ids)

        return ids, token_type, attention_mask
    
    def __len__(self):
        return sum([len(self.qrydoc_pair[query]) for query in self.qrydoc_pair])

    def __getitem__(self, idx):
        q_name = self.query_list[idx//self.doc_size]
        d_name = self.qrydoc_pair[q_name][idx%self.doc_size]
        
        if self.split == "Gen":
            with open(f'{self.train_dir}/query/{q_name}') as q:
                query = q.read()
            with open(f'{self.doc_dir}/{d_name}') as d:
                doc = d.read()
            
            ids, token_type, attention_mask = self._preprocess(query, doc)

            ids = torch.LongTensor(ids)
            token_type = torch.LongTensor(token_type)
            attention_mask = torch.LongTensor(attention_mask)
            return ids, token_type, attention_mask, q_name, d_name
        
        if self.split == "Dis":
            with open(f'{self.train_dir}/query/{q_name}') as q:
                query = q.read()
            with open(f'{self.doc_dir}/{d_name}') as d:
                gen_doc = d.read()

            gen_ids, gen_token_type, gen_attention_mask = self._preprocess(query, gen_doc)
            gen_ids = torch.LongTensor(gen_ids)
            gen_token_type = torch.LongTensor(gen_token_type)
            gen_attention_mask = torch.LongTensor(gen_attention_mask)

            pos_d_name = np.random.choice(self.pos[q_name])
            with open(f'{self.doc_dir}/{pos_d_name}') as pos_d:
                pos_doc = pos_d.read()
            
            pos_ids, pos_token_type, pos_attention_mask = self._preprocess(query, pos_doc)
            pos_ids = torch.LongTensor(pos_ids)
            pos_token_type = torch.LongTensor(pos_token_type)
            pos_attention_mask = torch.LongTensor(pos_attention_mask)

            return gen_ids, gen_token_type, gen_attention_mask, pos_ids, pos_token_type, pos_attention_mask
        
        if self.split == "Test":
            with open(f'{self.test_dir}/query/{q_name}') as q:
                query = q.read()
            with open(f'{self.doc_dir}/{d_name}') as d:
                doc = d.read()
            ids, token_type, attention_mask = self._preprocess(query, doc)

            ids = torch.LongTensor(ids)
            token_type = torch.LongTensor(token_type)
            attention_mask = torch.LongTensor(attention_mask)
            return ids, token_type, attention_mask, q_name, d_name


def get_pos_qrydoc_pair(train_dir):
    relevance_qrydoc_pair = {}
    with open(f"{train_dir}/pos.txt", 'r') as pos:
        for line in pos.readlines():
            if line.split()[0] in relevance_qrydoc_pair:
                values = relevance_qrydoc_pair[line.split()[0]]
                values.append(line.split()[2])
                relevance_qrydoc_pair[line.split()[0]] = values
            else:
                relevance_qrydoc_pair[line.split()[0]] = [line.split()[2]]

    return relevance_qrydoc_pair

def get_qrydoc_pair(dir):
    qrydoc_pair = {}
    with open("./data/doc_list.txt") as doc:
        doc_list = [d.rstrip() for d in doc.readlines()]

    with open(f"{dir}/query_list.txt") as query:
        query_list = [q.rstrip() for q in query.readlines()]

    for query in query_list:
        qrydoc_pair[query] = doc_list
    
    return qrydoc_pair     

def get_dataset(train_dir, test_dir, doc_dir, qrydoc_pair=None,split=None):
    assert split in ["All", "Topk", "Topk+Pos", "Test"]
    
    if split == "All":
        qrydoc_pair = get_qrydoc_pair(train_dir)
        all_dataset = BertDataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, qrydoc_pair=qrydoc_pair, split="Gen")
        return all_dataset

    if split == "Topk":
        topk_dataset = BertDataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, qrydoc_pair=qrydoc_pair, split="Gen")

        return topk_dataset

    if split == "Topk+Pos":
        pos = get_pos_qrydoc_pair(train_dir)
        dis_dataset = BertDataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, pos= pos,qrydoc_pair=qrydoc_pair, split="Dis")

        return dis_dataset
    
    if split == "Test":
        qrydoc_pair = get_qrydoc_pair(test_dir)
        test_dataset = BertDataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, qrydoc_pair=qrydoc_pair, split="Test")
        
        return test_dataset
from torch.utils.data import DataLoader
from torch import nn

from dataset import *

def gen_collate_fn(batch):
    ids, token_type, attention_mask, q_name, d_name = zip(*batch)
    ids = nn.utils.rnn.pad_sequence(ids, batch_first=True)
    token_type = nn.utils.rnn.pad_sequence(token_type, batch_first=True)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)

    return ids, token_type, attention_mask, q_name, d_name

def dis_collate_fn(batch):
    gen_ids, gen_token_type, gen_attention_mask, pos_ids, pos_token_type, pos_attention_mask = zip(*batch)
    gen_ids = nn.utils.rnn.pad_sequence(gen_ids, batch_first=True)
    gen_token_type = nn.utils.rnn.pad_sequence(gen_token_type, batch_first=True)
    gen_attention_mask = nn.utils.rnn.pad_sequence(gen_attention_mask, batch_first=True)
    pos_ids = nn.utils.rnn.pad_sequence(pos_ids, batch_first=True)
    pos_token_type = nn.utils.rnn.pad_sequence(pos_token_type, batch_first=True)
    pos_attention_mask = nn.utils.rnn.pad_sequence(pos_attention_mask, batch_first=True)

    return gen_ids, gen_token_type, gen_attention_mask, pos_ids, pos_token_type, pos_attention_mask

def get_dataloader(train_dir, test_dir, doc_dir, qrydoc_pair=None, batch_size=4, split=None):
    assert split in ["All", "Topk", "Topk+Pos", "Test"]
    
    if split == "All":
        all_dataset = get_dataset(train_dir, test_dir, doc_dir, split="All")
        all_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=gen_collate_fn)

        return all_dataloader

    if split == "Topk":
        topk_dataset = get_dataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, qrydoc_pair=qrydoc_pair, split="Topk")
        topk_dataloader = DataLoader(topk_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=gen_collate_fn)

        return topk_dataloader

    if split == "Topk+Pos":
        topk_pos_dataset = get_dataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, qrydoc_pair=qrydoc_pair, split="Topk+Pos")
        topk_pos_dataloader = DataLoader(topk_pos_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dis_collate_fn)

        return topk_pos_dataloader
    
    if split == "Test":
        tess_dataset = get_dataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, split="Test")
        test_dataloader = DataLoader(topk_pos_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=gen_collate_fn)
        
        return test_dataloader
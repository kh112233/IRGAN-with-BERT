from torch import nn

from transformers import *

class Generator(nn.Module):
    """
    IRGAN Generator aim to pick the high relevance score query-document pair.
    The document pick by generator will try to fool IRGAN Discriminator.
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        # Setting bert model for ranking the query-document pair
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        logits = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        return logits        

class Discriminator(nn.Module):
    """
    IRGAN Discriminator aim to find out which document is relevance document of query.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # Setting bert model for ranking the generaterd query-document pair and relevance query-document pair.
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1) 
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        logits = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        return logits   
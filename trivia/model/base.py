from torch import nn

from model.modeling import BertForQuestionAnswering
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class Base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = BertForQuestionAnswering.from_pretrained(cfg.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))

    def qa_forward(self, **batch):
        input_ids = batch['input_ids']
        input_mask = batch['input_mask']
        segment_ids = batch['segment_ids']
        s_idx = batch['s_idx']
        e_idx = batch['e_idx']
        loss, start_logits, end_logits = self.model.forward(input_ids=input_ids,
                                                            token_type_ids=segment_ids,
                                                            attention_mask=input_mask,
                                                            start_positions=s_idx,
                                                            end_positions=e_idx)
        return dict(loss=loss, start_logits=start_logits, end_logits=end_logits)

    def forward_ctx(self, input_ids, segment_ids):
        ctx = self.model.bert.embeddings(input_ids=input_ids, token_type_ids=segment_ids)
        return ctx

    def forward_ques(self, input_ids, segment_ids):
        ques = self.model.bert.embeddings(input_ids=input_ids, token_type_ids=segment_ids)
        return ques

    def mem_forward(self, **kwargs):
        raise NotImplementedError()

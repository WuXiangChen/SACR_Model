# 这里是codet5的基本代码，这里应该是直接对应着T5部分的内容
import os
from safetensors import safe_open
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    RobertaTokenizer)
import logging

logger = logging.getLogger(__name__)
# 这里可以模仿着codeReviewer对codet5进行设计
# 除了config基础配置信息不一样，其它的好像都没有特别的差别
class codet5Model(T5ForConditionalGeneration):
    def __init__(self, config, args=None):
        if config!=None:
            super().__init__(config)
            self.cls_head = nn.Linear(self.config.d_model, 2, bias=True)                     
            self.init()
  
        self.args = args
        self._init_model_classes()
        
    def init(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        factor = self.config.initializer_factor
        self.cls_head.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
        self.cls_head.bias.data.zero_()

    def _init_model_classes(self):
        """动态设置模型相关类"""
        # 默认使用T5系列组件
        self.config_class = getattr(self.args, 'config_class', T5Config)
        self.model_class = getattr(self.args, 'model_class', codet5Model)
        self.tokenizer_class = getattr(self.args, 'tokenizer_class', RobertaTokenizer)
    
    def forward(self, *argv, **kwargs):
        if "cls" in kwargs:
            assert ("input_ids" in kwargs and "labels" in kwargs and "attention_mask" in kwargs)
            return self.cls(input_ids=kwargs["input_ids"], labels=kwargs["labels"], attention_mask=kwargs["attention_mask"])
        
        if "input_labels" in kwargs:
            assert ("input_ids" in kwargs and "input_labels" in kwargs and "decoder_input_ids" in kwargs and "attention_mask" in kwargs and "decoder_attention_mask" in kwargs),\
            "Please give these arg keys."
            input_ids = kwargs["input_ids"]
            input_labels = kwargs["input_labels"]
            decoder_input_ids = kwargs["decoder_input_ids"]
            attention_mask = kwargs["attention_mask"]
            decoder_attention_mask = kwargs["decoder_attention_mask"]
            if "encoder_loss" not in kwargs:
                encoder_loss = True
            else:
                encoder_loss = kwargs["encoder_loss"]
            return self.review_forward(input_ids, input_labels, decoder_input_ids, attention_mask, decoder_attention_mask, encoder_loss)
        return super().forward(*argv, **kwargs)

    # 做cls的话 就只使用encoder就可以了
    def cls(self, input_ids, labels, attention_mask):
        # print("AAAAAAAAAAAA")
        # print(f"Max input_id: {torch.max(input_ids)}")
        # real_vocab_size = self.get_input_embeddings().weight.shape[0]
        # print(f"Vocab size: {real_vocab_size}")
        assert torch.all(input_ids < self.encoder.config.vocab_size), "Input IDs contain out-of-vocabulary tokens"
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, return_dict=False)
        hidden_states = encoder_outputs[0]
        first_hidden = hidden_states[:, 0, :]
        first_hidden = nn.Dropout(0.3)(first_hidden)
        logits = self.cls_head(first_hidden)
        loss_fct = CrossEntropyLoss()
        if labels != None:
            loss = loss_fct(logits, labels)
            # logger.info(f"DONE FOR TRAINING Logits:{logits}")
            return loss
        return logits

    def review_forward(
        self,
        input_ids,
        input_labels,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        encoder_loss=True
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False)
        
        hidden_states = encoder_outputs[0] # 取张量的第一个元素，作为hidden_state
        decoder_inputs = self._shift_right(decoder_input_ids)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False)
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings: # this is True default
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        if encoder_loss:
            # print(self.encoder.get_input_embeddings().weight.shape)
            cls_logits = nn.functional.linear(hidden_states, self.encoder.get_input_embeddings().weight)
            # cls_logits = self.cls_head(hidden_states)
        lm_logits = self.lm_head(sequence_output)
        if decoder_input_ids is not None:
            lm_loss_fct = CrossEntropyLoss(ignore_index=0)      # Warning: PAD_ID should be 0
            loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_input_ids.view(-1))
            if encoder_loss and input_labels is not None:
                cls_loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss += cls_loss_fct(cls_logits.view(-1, cls_logits.size(-1)), input_labels.view(-1))
            return loss
        return cls_logits, lm_logits


import os
import json
import torch
from torch.utils.data import DistributedSampler
from base_trainer import BaseTrainer
from utils import RefineDataset, SimpleRefineDataset
from .smooth_bleu import bleu_fromstr
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch

class RefinementTrainer(BaseTrainer):
    def get_data_loader(self, data_file, eval=False):
        # 初始化数据集
        dataset_cls = SimpleRefineDataset if self.args.raw_input else RefineDataset
        dataset = dataset_cls(
            self.tokenizer, 
            self.pool, 
            self.args, 
            data_file
        )
        
        # 创建采样器
        sampler = DistributedSampler(dataset) if not eval else SequentialSampler(dataset)
        
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.cpu_count,
            collate_fn=lambda x: x
        )

    def train_step(self, examples):
        # 准备输入数据
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], 
            dtype=torch.long
        ).to(self.local_rank)
        
        target_ids = torch.tensor(
            [ex.target_ids for ex in examples],
            dtype=torch.long
        ).to(self.local_rank)
        
        # 计算损失
        outputs = self.model(
            input_ids=source_ids,
            decoder_input_ids=target_ids[:, :-1],  # 左移处理
            labels=target_ids[:, 1:],  # 右移处理
            encoder_loss=False
        )
        return outputs.loss

    def evaluate(self, dataloader):
        self.model.eval()
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
            
        pred_nls, golds = [], []
        for examples in dataloader:
            # 生成预测
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], 
                dtype=torch.long
            ).to(self.local_rank)
            
            preds = model
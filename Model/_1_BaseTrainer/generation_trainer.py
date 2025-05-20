import torch
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from .base_trainer import BaseTrainer
from .utils import CommentGenDataset, SimpleGenDataset
from .smooth_bleu import bleu_fromstr

class GenerationTrainer(BaseTrainer):
    def get_data_loader(self, data_file, eval=False):
        # 初始化数据集
        dataset_cls = SimpleGenDataset if self.args.raw_input else CommentGenDataset
        dataset = dataset_cls(self.tokenizer, self.pool, self.args, data_file) # 这里数据集的信息应当也是共有的，包括cls数据；msg数据；以及ref数据
        # 创建采样器
        sampler = DistributedSampler(dataset) if not eval else SequentialSampler(dataset)
        # 创建DataLoader
        return DataLoader(dataset,sampler=sampler,batch_size=self.args.eval_batch_size if eval else self.args.train_batch_size,
            num_workers=self.args.cpu_count, collate_fn=lambda x: x) # 这部分应该是共用的

    def train_step(self, examples):
        # 准备输入数据
        source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(self.local_rank)
        target_ids = torch.tensor([ex.target_ids for ex in examples], dtype=torch.long).to(self.local_rank)
        # 计算损失
        outputs = self.model(input_ids=source_ids, decoder_input_ids=target_ids[:, :-1], labels=target_ids[:, 1:], encoder_loss=False)
        return outputs.loss

    def evaluate(self, dataloader):
        self.model.eval()
        # 首先，这可能和PyTorch的模型并行或数据并行有关。比如，当使用DataParallel或者DistributedDataParallel时，模型会被包裹一层，原来的模型会被放在module属性里。
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
            
        pred_nls, golds = [], []
        for examples in dataloader:
            # 生成预测
            source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(self.local_rank)
            preds = model.generate(
                source_ids,
                attention_mask=source_ids.ne(self.tokenizer.pad_id),
                use_cache=True,
                num_beams=self.args.beam_size,
                early_stopping=True,
                max_length=self.args.max_target_length)
            
            # 解码预测结果
            batch_preds = [self.tokenizer.decode(
                ids[1:],  # 跳过起始token
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False) for ids in preds.cpu().numpy()]
  
            pred_nls.extend(batch_preds)
            golds.extend([ex.msg for ex in examples])

        # 计算BLEU
        return bleu_fromstr(pred_nls, golds, rmstop=False)

    def save_model(self, output_dir, metric_value):
        super().save_model(output_dir, metric_value)
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
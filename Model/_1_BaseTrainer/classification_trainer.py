# classification_trainer.py
import os
import pdb
from Model._1_BaseTrainer.configs import set_seed
from .base_trainer import BaseTrainer
from .utils import CommentClsDataset, SimpleClsDataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationTrainer(BaseTrainer):
    def  __init__(self, args, data_file, model, eval_=False):
        super().__init__(args=args, model=model)
        self.args = args
        self.data_file = data_file
        self.eval_ = eval_

    def get_data_loader(self, train_eval_=False):
        print(self.data_file)
        def fn(features):
            return features
        if not train_eval_ and os.path.isdir(self.data_file):
            train_files = [file for file in os.listdir(self.data_file) if file.startswith("cls-train-chunk") and file.endswith(".jsonl")]
            train_files = [os.path.join(self.data_file, file) for file in train_files]
        elif train_eval_ and os.path.isdir(self.data_file):
            train_files = [file for file in os.listdir(self.data_file) if file.startswith("cls-valid") and file.endswith(".jsonl")]
            train_files = [os.path.join(self.data_file, file) for file in train_files]
        else:# 这里对应着test的过程
            train_files = [self.data_file]
        
          # 合并所有文件的数据
        all_datasets = []
        for data_file in train_files:
            dataset = (SimpleClsDataset if self.args.raw_input else CommentClsDataset)(
                self.tokenizer, self.pool, self.args, data_file)
            all_datasets.append(dataset)
        
        # 使用ConcatDataset合并所有数据集
        combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
        
        # 创建DataLoader
        sampler = SequentialSampler(combined_dataset) if self.eval_ else DistributedSampler(combined_dataset)
        dataloader = DataLoader(
            combined_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size if self.eval_ else self.args.train_batch_size,
            num_workers=self.args.cpu_count,
            collate_fn=fn)
        return dataloader

    def train_step(self, examples):
        source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(self.local_rank)
        ys = torch.tensor([ex.y for ex in examples], dtype=torch.long).to(self.local_rank)
        source_mask = source_ids.ne(self.tokenizer.pad_id)
        loss = self.model(cls=True, input_ids=source_ids, labels=ys, attention_mask=source_mask)
        return loss

    def evaluate(self, dataloader):
        self.model.eval()
        pred, gold = [], []
        with torch.no_grad():
            for examples in dataloader:
                source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(self.local_rank)
                source_mask = source_ids.ne(self.tokenizer.pad_id)
                logits = self.model(cls=True, input_ids=source_ids, labels=None, attention_mask=source_mask).to(self.local_rank)
                pred.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                gold.extend([ex.y for ex in examples])
        acc = accuracy_score(gold, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(gold, pred, average='binary', zero_division=0)
        # 打印混淆矩阵
        cm = confusion_matrix(gold, pred)
        logger.info(f"Confusion Matrix:{cm}")
        logger.info(f"Eval Results - ACC: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return acc, precision, recall, f1
    
    def run(self):
        try:
            # Initialize training
            global_step = 0
            tr_loss, logging_loss = 0.0, 0.0
            best_f1 = 0.0
            train_dataloader = self.get_data_loader()
            eval_dataloader  = self.get_data_loader(train_eval_=True)
            # Training loop
            for epoch in range(1, self.args.train_epochs + 1):
                # Set seed for reproducible data split
                save_seed = self.args.seed
                self.args.seed += epoch
                set_seed(self.args)
                self.args.seed = save_seed
                self.model.train()
                for step, examples in enumerate(train_dataloader, 1):
                    if step == 1:
                        # ex = examples[0]
                        logger.info(f"batch size: {len(examples)}")
                        # logger.info(f"example source: {tokenizer.convert_ids_to_tokens(ex.source_ids)}")
                    step+=1
                    # print("examples:",type(examples))
                    loss = self.train_step(examples)
                    if self.args.gpu_per_node > 1:
                        loss = loss.mean()
                    
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    
                    loss.backward()
                    tr_loss += loss.item()
                    
                    # Gradient accumulation and parameter update
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                        
                        # Logging
                        if self.args.global_rank == 0 and global_step % self.args.log_steps == 0:
                            current_loss = (tr_loss - logging_loss) / self.args.log_steps
                            logger.info(
                                f"Epoch: {epoch}, Step: {global_step}/{self.args.train_steps}, "
                                f"Loss: {current_loss:.4f}")
                            logging_loss = tr_loss
                        
                        # Evaluation and model saving
                        if self.args.global_rank == 0 and (
                            global_step % self.args.save_steps == 0 or 
                            global_step == self.args.train_steps):
                            acc, precision, recall, f1 = self.evaluate(eval_dataloader)
                            logger.info(f"Validation Accuracy at step {global_step}: {f1:.4f}")
                            
                            # Save checkpoint
                            if f1 > best_f1:
                                best_f1 = f1
                                output_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}-{f1:.4f}")
                                os.makedirs(output_dir, exist_ok=True)
                                self.save_model(output_dir, f1)
                                logger.info(f"New best model saved to {output_dir}")
                    
                    # Early stopping if reach max steps
                    if global_step >= self.args.train_steps:
                        if self.args.global_rank == 0:
                            f1, precision, recall, f1 = self.evaluate(eval_dataloader)
                            output_dir = os.path.join(
                                self.args.output_dir, 
                                f"checkpoint-last-{f1:.4f}")
                            self.save_model(output_dir, f1)
                            logger.info(f"Training completed. Final model saved to {output_dir}")
                        return
        finally:
            self.pool.close()
            if self.args.global_rank == 0:
                logger.info("Training finished. Cleaning up resources.")
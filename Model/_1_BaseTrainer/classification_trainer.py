# classification_trainer.py
import os
from _1_BaseTrainer.configs import set_seed
from base_trainer import BaseTrainer
from utils import CommentClsDataset, SimpleClsDataset
from sklearn.metrics import accuracy_score
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
    def get_data_loader(self, data_file, eval_=False):
        dataset = (SimpleClsDataset if self.args.raw_input else CommentClsDataset)(
            self.tokenizer, self.pool, self.args, data_file)
        sampler = SequentialSampler(dataset) if eval_ else DistributedSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=self.args.eval_batch_size if eval_ else self.args.train_batch_size)

    def train_step(self, examples):
        source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(self.local_rank)
        ys = torch.tensor([ex.y for ex in examples], dtype=torch.long).to(self.local_rank)
        source_mask = source_ids.ne(self.tokenizer.pad_id)
        outputs = self.model(cls=True, input_ids=source_ids, labels=ys, attention_mask=source_mask)
        return outputs.loss

    def evaluate(self, dataloader):
        self.model.eval()
        pred, gold = [], []
        with torch.no_grad():
            for examples in dataloader:
                logits = self.model(cls=True, input_ids=torch.tensor([ex.source_ids for ex in examples]).to(self.local_rank))
                pred.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                gold.extend([ex.y for ex in examples])
        return accuracy_score(gold, pred)
    
    def run(self):
        try:
            # Initialize training
            global_step = 0
            tr_loss, logging_loss = 0.0, 0.0
            best_acc = 0.0
            
            # Prepare data loaders
            train_dataloader = self.get_data_loader(self.args.train_filename)
            eval_dataloader = self.get_data_loader(self.args.dev_filename, eval_=True)
            
            # Training loop
            for epoch in range(1, self.args.num_train_epochs + 1):
                # Set seed for reproducible data split
                save_seed = self.args.seed
                self.args.seed += epoch
                set_seed(self.args)
                self.args.seed = save_seed
                
                self.model.train()
                for step, examples in enumerate(train_dataloader, 1):
                    # Training step
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
                                f"Loss: {current_loss:.4f}"
                            )
                            logging_loss = tr_loss
                        
                        # Evaluation and model saving
                        if self.args.global_rank == 0 and (
                            global_step % self.args.save_steps == 0 or 
                            global_step == self.args.train_steps
                        ):
                            acc = self.evaluate(eval_dataloader)
                            logger.info(f"Validation Accuracy at step {global_step}: {acc:.4f}")
                            
                            # Save checkpoint
                            if acc > best_acc:
                                best_acc = acc
                                output_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}-{acc:.4f}")
                                os.makedirs(output_dir, exist_ok=True)
                                self.save_model(output_dir, acc)
                                logger.info(f"New best model saved to {output_dir}")
                    
                    # Early stopping if reach max steps
                    if global_step >= self.args.train_steps:
                        if self.args.global_rank == 0:
                            acc = self.evaluate(eval_dataloader)
                            output_dir = os.path.join(
                                self.args.output_dir, 
                                f"checkpoint-last-{acc:.4f}"
                            )
                            self.save_model(output_dir, acc)
                            logger.info(f"Training completed. Final model saved to {output_dir}")
                        return
        finally:
            self.pool.close()
            if self.args.global_rank == 0:
                logger.info("Training finished. Cleaning up resources.")
# classification_trainer.py
from base_trainer import BaseTrainer
from utils import CommentClsDataset, SimpleClsDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch

class ClassificationTrainer(BaseTrainer):
    def get_data_loader(self, data_file, eval=False):
        dataset = (SimpleClsDataset if self.args.raw_input else CommentClsDataset)(
            self.tokenizer, self.pool, self.args, data_file
        )
        sampler = SequentialSampler(dataset) if eval else DistributedSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=self.args.eval_batch_size if eval else self.args.train_batch_size)

    def train_step(self, examples):
        source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(self.local_rank)
        ys = torch.tensor([ex.y for ex in examples], dtype=torch.long).to(self.local_rank)
        outputs = self.model(cls=True, input_ids=source_ids, labels=ys)
        return outputs.loss

    def evaluate(self, dataloader):
        pred, gold = [], []
        for examples in dataloader:
            logits = self.model(cls=True, input_ids=torch.tensor([ex.source_ids for ex in examples]).to(self.local_rank))
            pred.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            gold.extend([ex.y for ex in examples])
        return accuracy_score(gold, pred)
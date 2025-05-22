# 这里我希望继承BaseTrainer来控制多卡训练框架


'''
  导包区
'''
import logging
from Model._1_BaseTrainer import ClassificationTrainer, GenerationTrainer, RefinementTrainer
from transformers import get_scheduler
from torch.optim import AdamW

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO)
logger = logging.getLogger(__name__)

## 分别以CodeReviewer的基本代码 实现三个需求逻辑
# 这里只是复用 Trainer的过程，还不是Model的定义过程
class t5crCLS(ClassificationTrainer):
  def __init__(self, args, data_file: str, model=None,  eval_=False):
    super().__init__(args=args, data_file=data_file, model=model, eval_=eval_)
  
  def evaluate(self, data_file:str=None):
    if data_file!=None:
      self.data_file = data_file
      dataloader = self.get_data_loader(train_eval_=True)
      return super().evaluate(dataloader)  
    else:
      raise ValueError("No data_file provided for evaluation.")

  # 这里可以独立的设计 _setup_training, 来初始化它自己的optimizer和scheduler的设计
  def _setup_training(self,
                      scheduler_type="polynomial", 
                      num_warmup_steps=10000, 
                      num_training_steps=500):
    # 参数分组    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]
    self.optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    # 学习率调度器
    self.scheduler = get_scheduler(
      scheduler_type, 
      optimizer=self.optimizer, 
      num_warmup_steps=num_warmup_steps, 
      num_training_steps=num_training_steps)
  def run(self):
    return super().run()

class t5crMSG(GenerationTrainer):
  pass

class t5crREF(RefinementTrainer):
  pass
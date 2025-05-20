# 本节的主要目标是分别继承三个任务的基本模型，然后提供一个运行的接口？
## 不 运行的接口统一放在main中进行主动调用


'''
  导包区
'''
import logging
from Model._1_BaseTrainer import ClassificationTrainer, GenerationTrainer, RefinementTrainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)

## 分别以CodeReviewer的基本代码 实现三个需求逻辑
# 这里只是复用 Trainer的过程，还不是Model的定义过程
class CodeReviewerCLS(ClassificationTrainer):
  def __init__(self, args, datafile: str, model=None,  eval_=False):
    super().__init__(args=args, datafile=datafile, model=model, eval_=eval_)
  
  def run(self):
    return super().run()

class CodeReviewerMSG(GenerationTrainer):
  pass

class CodeReviewerREF(RefinementTrainer):
  pass
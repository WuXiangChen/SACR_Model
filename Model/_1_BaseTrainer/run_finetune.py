from classification_trainer import ClassificationTrainer
from generation_trainer import GenerationTrainer  # 类似分类的实现
from refinement_trainer import RefinementTrainer  # 类似分类的实现

def run_with_args(args):
    trainers = {
        "cls": ClassificationTrainer,
        "msg": GenerationTrainer,
        "ref": RefinementTrainer}
    trainer = trainers[args.task_type](args)
    trainer.run()
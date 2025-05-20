import argparse
from classification_trainer import ClassificationTrainer
from generation_trainer import GenerationTrainer  # 类似分类的实现
from refinement_trainer import RefinementTrainer  # 类似分类的实现

def add_args(parser):
    # 公共参数
    parser.add_argument("--task_type", choices=["cls", "msg", "ref"], required=True)
    # 添加各任务特有参数...
    return parser

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    
    trainers = {
        "cls": ClassificationTrainer,
        "msg": GenerationTrainer,
        "ref": RefinementTrainer
    }
    
    trainer = trainers[args.task_type](args)
    trainer.run()

if __name__ == "__main__":
    main()
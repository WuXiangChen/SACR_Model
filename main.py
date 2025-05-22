import logging
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 必须在导入 torch 前设置！
from Model import *
from config import get_args
from general_LLMs_hooker import QAProcessor


logger = logging.getLogger(__name__)

if __name__ == "__main__":
  args = get_args()
  ##########################
  print("="*50)
  print("Program Starting with Parameters:")
  if args.general_model:
    print(f"Model Name: {args.model_name}")
  else:
    print(f"Model Name: {args.model_type}")
  print(f"Dataset Name: {args.dataset_name}")
  print("="*50)
  print()
  ##########################

  # 做程序入口

  # 确定数据集的加载路径
  datafile_path = f"../ACR_Dataset/{args.dataset_name}/{args.task_type}"
  
  g_bool = args.general_model
  result_output_folder = f"Results/{args.dataset_name}/{args.task_type}"
  os.makedirs(result_output_folder, exist_ok=True)

  if g_bool:
    processor = QAProcessor(model_name=args.model_name, dataset_name=args.dataset_name, max_workers=4, save_interval=10)
    processor.process_dataset(data_path=datafile_path, sample_size=200)
  else:
    configPath = f"./Model/{args.model_type}"
    config = os.path.join(configPath, "config.json")
    model = eval(f"{args.model_type}Model")(args=args, config=None)
    # 将模型注入到训练过程中
    if args.train_eval:
      args.model_name_or_path = f"../ACR_Model_Saved/{args.model_type}/originalModel/" # 这里集中了模型信息加载的基本内容，包括config、base_model
      args.output_dir = f"../ACR_Model_Saved/{args.model_type}/{args.task_type}/"
      os.makedirs(args.output_dir, exist_ok=True)

      args.dev_filename = f"../ACR_Dataset/{args.dataset_name}/{args.task_type}/{args.task_type}-valid.jsonl"
      args.train_filename = f"../ACR_Dataset/{args.dataset_name}/{args.task_type}/"
      
      logger.info(f"Training/eval parameters: model_name_or_path={args.model_name_or_path}, output_dir={args.output_dir}, dev_filename={args.dev_filename}, train_filename={args.train_filename}")

      trainer = eval(f"{args.model_type}{args.task_type.upper()}")(args=args, data_file=args.train_filename, model=model, eval_=False)
      trainer.run()

    else:
      # 测试
      args.test_filename = f"../ACR_Dataset/{args.dataset_name}/{args.task_type}/{args.task_type}-test.jsonl"
      trainer = eval(f"{args.model_type}{args.task_type}")(args=args, data_file=args.test_filename, model=model, eval_=True)
      re = trainer.evaluate()
      '''
        这里缺少评估、指标和对结果的整理
      '''

  # Create Results directory if it doesn't exist
  print("&"*50)
  print(f"Final Model Name: {args.model_type}")
  print(f"Final Dataset Name: {args.dataset_name}")
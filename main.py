from typing import get_args
from Utils.evaluation import MetricsEvaluator
import os
from Model import *
from general_LLMs_hooker import QAProcessor


if __name__ == "__main__":
  args = get_args()
  ##########################
  print("="*50)
  print("Program Starting with Parameters:")
  print(f"Model Name: {args.model_name}")
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
    model = eval(f"{args.model_type}Model")(config)
    # 将模型注入到训练过程中
    ## 增加数据集的路径信息
    if args.eval_NE:
      args.model_name_or_path = f"../ACR_Model_Saved/{args.model_type}/originalModel/"
      args.output_dir = f"../ACR_Model_Saved/{args.model_type}/{args.task_type}/"
      args.train_filename = f"../ACR_Dataset/{args.model_type}/{args.task_type}/"
      args.dev_filename = f"../ACR_Dataset/{args.model_type}/{args.task_type}/{args.task_type}-valid.jsonl"
      trainer = eval(f"{args.model_type}{args.task_type}")(args=args, datafile=args.train_filename, model=model, eval_=False)
    else:
      args.datafilePath = f"{args.dataset_name}/{args.task_type}/"
      trainer = eval(f"{args.model_type}{args.task_type}")(args=args, datafile=args.datafilePath, model=model, eval_=True)

  # Create Results directory if it doesn't exist
  print(f"Model Name: {args.model_type}")
  print(f"Dataset Name: {args.dataset_name}")
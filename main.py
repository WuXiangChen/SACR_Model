import argparse
import datetime
from tqdm import tqdm
from Utils.data_util import JSONLReader, save_results
from Model.generalLLMs.remote_server import DeepSeekClient
from Utils.evaluation import MetricsEvaluator
import os
import json
import concurrent.futures
from tqdm import tqdm
import queue

from general_LLMs_hooker import QAProcessor


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="The entrence function for S-ACR process.")
  parser.add_argument("-m", "--model_name", type=str, choices=["gpt_4o", "ds_671B", "ds_reasoner", "qwen_2.5", "llama_3.1"], default="qwen_2.5",help="The model to be used (options: gpt_4o, ds_671B, qwen_2.5, llama_3.1)")
  parser.add_argument("-d", "--dataset_name", type=str, choices=["CR", "CarLLM", "T5CR"], default="CR", help="The dataset given to be used")
  parser.add_argument("-g", "--general_model", type=bool, default=True, help="The general LLMs are used, rather than dedicated")
  ########### 以上是为，通用模型的模型参数选择；以下是为专用模型的参数选择 ###########
  parser.add_argument("-ms", "--model_specific_name", type=str, choices=["codereviewer", "t5cr", "codefinder", "llaMa_reviewer", "codedoctor", "codeT5_shepherd", "inferFix", "auger","jLED", "DAC"])
  parser.add_argument("-ts", "--task_type", type=str, choices=["cls", "msg", "ref"], required=True)
  args = parser.parse_args()
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
    # 这里根据不同的dedicated model name进行eval不同模型
    pass
    

  # Create Results directory if it doesn't exist
  print(f"Model Name: {args.model_name}")
  print(f"Dataset Name: {args.dataset_name}")
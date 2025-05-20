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

def process_qa(qa, cli, results_queue):
    input_ = qa["input"]
    while True:
        try:
            output = cli.perform_code_review(input_)
            # 直接将整个output作为字符串保存
            results_queue.put({
                "Question": input_,
                "RawOutput": str(output)  # 确保转换为字符串
            })
            break
        except Exception as e:
            print(f"Error occurred during processing: {e}. Retrying...")
            continue


def process_concurrently(QAs, cli, args, save_interval=5, max_workers=1):
    results_queue = queue.Queue()
    processed_count = 0
    
    # Process QAs concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for qa in QAs:
            futures.append(executor.submit(
                process_qa,
                qa=qa,
                cli=cli,
                results_queue=results_queue
            ))
        
        # Progress bar
        for _ in tqdm(concurrent.futures.as_completed(futures), 
                     total=len(QAs), 
                     desc="Processing QAs"):
            processed_count += 1
            
            # Save every save_interval iterations
            if processed_count % save_interval == 0:
                save_results(results_queue, model_name=args.model_name, dataset_name=args.dataset_name)
        
        # Save any remaining results at the end
        if processed_count % save_interval != 0:
            results_queue.put(None)
            save_results(results_queue, model_name=args.model_name, dataset_name=args.dataset_name)


def process(data_path:str = None):
  if data_path==None:
    raise "Please give the proper data file path"
  
  jsReader = JSONLReader(data_path)
  count_len =jsReader.count_lines()
  QAs = jsReader.read_lines(start=count_len//2, end=count_len//2+200)
  cli = DeepSeekClient(base_model=args.model_name)
  # 并发处理所有的问题
  process_concurrently(QAs, cli, args)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="The entrence function for S-ACR process.")
  parser.add_argument("-m", "--model_name", type=str,choices=["gpt_4o", "ds_671B", "ds_reasoner", "qwen_2.5", "llama_3.1"], default="qwen_2.5",help="The model to be used (options: gpt_4o, ds_671B, qwen_2.5, llama_3.1)")
  parser.add_argument("-d", "--dataset_name", type=str, default="carllm", help="The dataset given to be used")

  ########### 以上是为，通用模型的模型参数选择；以下是为专用模型的参数选择 ###########
  parser.add_argument("-ms", "--model_specific_name", type=str, choices=["codereviewer", "t5cr", "codefinder", "llaMa_reviewer", "codedoctor", "codeT5_shepherd", "inferFix", "auger","jLED", "DAC"])


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
  if args.dataset_name == "carllm":
    datafile_path = "./Dataset/CarLLM-Data.jsonl"
  elif args.dataset_name == "other_dataset":
    datafile_path = "./Dataset/Other-Data.jsonl"
  else:
    raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")
  
  # Create Results directory if it doesn't exist
  os.makedirs("Results", exist_ok=True)
  process(data_path = datafile_path)
  print(f"Model Name: {args.model_name}")
  print(f"Dataset Name: {args.dataset_name}")
import argparse
from tqdm import tqdm
from Utils.data_util import JSONLReader
from Model.remote_server import DeepSeekClient
from Utils.evaluation import MetricsEvaluator
import os
import pandas as pd
import json
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import queue

def process_qa(qa, cli, results_queue):
    input_ = qa["input"]
    while True:
        try:
            output = cli.perform_code_review(input_)
            if "deficiency_existence" in output:
                results_queue.put({
                    "Question": input_,
                    "deficiency_existence": output["deficiency_existence"],
                    "code_review_suggestion": output["code_review_suggestion"],
                    "suggested_code": output["suggested_code"]
                })
                break
        except Exception as e:
            print("Error occurred during processing. Retrying...")
            print(e)
            print("Error occurred during processing. Retrying...")
            continue

def save_results(results_queue, model_name, dataset_name, save_interval=20):
    results = []
    counter = 0
    while True:
        try:
            item = results_queue.get(timeout=30)  # Timeout for safety
            results.append(item)
            counter += 1
            
            if counter % save_interval == 0:
                temp_output_file = f"Results/{model_name}_{dataset_name}.xlsx"
                df = pd.DataFrame(results)
                df.to_excel(temp_output_file, index=False)
                
        except queue.Empty:
            # Final save if queue is empty and no more items expected
            if results:
                temp_output_file = f"Results/{model_name}_{dataset_name}.xlsx"
                df = pd.DataFrame(results)
                df.to_excel(temp_output_file, index=False)
            break

def process_concurrently(QAs, cli, args, save_interval=20, max_workers=1):
    results_queue = queue.Queue()
    # Start the saver thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as saver_executor:
        saver_future = saver_executor.submit(
            save_results, 
            results_queue, 
            args.model_name, 
            args.dataset_name, 
            save_interval
        )
        
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
                pass

        # Signal saver to finish
        results_queue.put(None)
        concurrent.futures.wait([saver_future])

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
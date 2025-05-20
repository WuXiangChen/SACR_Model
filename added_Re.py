import os
import pandas as pd
import json
import ast
from Utils.data_util import JSONLReader

# 首先将所有的xlsx转成json文件
def convert_xlsx_to_json(directory):
  for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
      xlsx_path = os.path.join(directory, filename)
      json_path = os.path.join(directory, filename.replace(".xlsx", ".json"))
      
      # 读取xlsx文件
      data = pd.read_excel(xlsx_path)
      # 转换为json并保存
      data.to_json(json_path, orient="records", force_ascii=False, indent=4)
      print(f"Converted {xlsx_path} to {json_path}")

def merge_and_store_data(QAs, result_path, output_folder_path):
  merged_data = []
  result_files = [f for f in os.listdir(result_path) if f.endswith(".json")]
  result_files.sort()  # 确保按序处理

  for k, result_file in enumerate(result_files):
    result_data = []
    result_file_path = os.path.join(result_path, result_files[k])
    with open(result_file_path, "r", encoding="utf-8") as f:
      result_data = json.load(f)

    for i, qa in enumerate(QAs):
      if i < len(result_data):
        if "output" in qa:
          output_str = qa["output"].replace("false", "False").replace("true", "True")
          output = ast.literal_eval(output_str)
          re_line = result_data[i]
          merged_dict = {**re_line, **output}
          merged_data.append(merged_dict)
        else:
          print(f"Skipping QA at index {i} due to missing or invalid 'output' field.")
      else:
        break

    # 将合并后的数据存储到指定文件
    output_file = os.path.join(output_folder_path, result_file)
    with open(output_file, "w", encoding="utf-8") as f:
      json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"Merged data has been stored in {output_file}")


if __name__ == "__main__":
  # 调用函数
  result_path = "./Results/"
  # convert_xlsx_to_json(result_path)

  datafile_path = "./Dataset/CarLLM-Data.jsonl"
  jsReader = JSONLReader(datafile_path)
  count_len =jsReader.count_lines()
  QAs = jsReader.read_lines(start=count_len//2, end=count_len//2+200)
  # 将这个list中每个dict的output下的字典 与 result_path中的每个字典 按序合并
  # 调用方法
  output_folder_path = "./merged_Re/"
  merge_and_store_data(QAs, result_path, output_folder_path)
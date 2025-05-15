import os
import pandas as pd
import json

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

if __name__ == "__main__":
  # 调用函数
  result_path = "./Results/"
  # convert_xlsx_to_json(result_path)

  datafile_path = "./Dataset/CarLLM-Data.jsonl"
  jsReader = JSONLReader(datafile_path)
  count_len =jsReader.count_lines()
  QAs = jsReader.read_lines(start=count_len//2, end=count_len//2+200)
  print()
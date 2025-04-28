import json
import requests
from openai import OpenAI
from .CONSTANT import *
# 25.4.22 本节的主要目标，利用本地服务（DS系列）实现S-ACR的基本任务

class DeepSeekClient:
  def __init__(self, base_model="ds_671B"):
    if base_model in list(BASE_MODEL.keys()):
       api_key = BASE_MODEL[base_model]["API_KEY"]
       base_url = BASE_MODEL[base_model]["BASE_URL"]
       self.model_name = BASE_MODEL[base_model]["MODEL_NAME"]
    else:
        raise ValueError(f"Invalid base_model '{base_model}' provided. Please choose from: {', '.join(BASE_MODEL.keys())}")

    self.client = OpenAI(
        api_key=f"{api_key}",
        base_url=base_url,
    )
        
  def perform_code_review(self, code_diff, additional_context=None):
    system_prompt = """
      You are an AI code reviewer with expertise in Python programming. Your task is to analyze code changes (Code Diff) and provide a comprehensive review in JSON format. Specifically, you should:
      1. Identify potential issues like bugs, inefficiencies, or violations of best practices
      2. Suggest improvements for readability, maintainability, and performance
      3. Highlight any followed best practices
      4. Provide a revised version of the code
      5. Ensure your response is complete and detailed and make the return reponse as json type

      Respond in this JSON format:
      {
          "deficiency_existence": "Yes or No",
          "code_review_suggestion": "Your detailed review here",
          "suggested_code": "Your improved code here"
      }

      EXAMPLE INPUT:
      - def cal(lst):
      -     s = 0
      -     for i in lst:
      -         s += i
      -     return s
      + def calculate_sum(numbers):
      +     sum = 0
      +     for n in numbers:
      +         sum += n
      +     return sum

      EXAMPLE JSON OUTPUT:
      {
          "deficiency_existence": "Yes",
          "code_review_suggestion": "The code could use more meaningful variable names. Consider using Python's built-in sum() function for simplicity.",
          "suggested_code": "def calculate_sum(nums):\n    try:\n        return sum(nums)\n    except TypeError:\n        print(\"Error: Input must be iterable of numbers\")\n        return 0"
      }

      """
    messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": code_diff}]

    try:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # response_format={'type': 'json_object'}
        )
        data = response.choices[0].message.content
        data = data.strip("```").strip("json").strip()
        # return json.loads(data)
        return data
    except Exception as e:
        print(f"Error during request: {e}")
        return None

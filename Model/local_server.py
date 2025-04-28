import requests
from .CONSTANT import *
# 25.4.22 本节的主要目标，利用本地服务（DS系列）实现S-ACR的基本任务

class DeepSeekClient:
  def __init__(self):
    self.base_url = f"http://{server_address}:{port}"
    self.auth = (username, password)

  def perform_code_review(self, code_diff, additional_context=None):
    endpoint = f"{self.base_url}/api/chat/"
    
    prompt = f'''
    You are an AI code reviewer with expertise in Python programming. Your task is to analyze the following code changes (Code Diff) and provide a comprehensive review. Specifically, you should:
    1. Identify potential issues, such as bugs, inefficiencies, or violations of best practices, in the proposed changes.
    2. Suggest improvements to enhance code readability, maintainability, and performance.
    3. Highlight any best practices that are followed in the changes and provide constructive feedback.
    4. Provide a revised version of the code that incorporates your recommendations.
    5. Ensure that no part of the code is omitted in your review or suggested revisions. Provide a complete and detailed response.

    Code Diff:
    {code_diff}

    Please provide your response in the following format:
    - Deficiency Existence:
      (Yes or No)
    - Code Review Suggestion:
      (Your detailed review here)
    - Suggested Code:
      (Your improved code here)
    '''

    template_answer = """
    Case 1:
    Give the code diff as follows, please generate code review and suggestions:
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

    - Deficiency Existence:
      - Yes

    Code Review Suggestion:
    - The code could use more meaningful variable names to improve readability, like renaming `lst` to `nums` and `s` to `total` so their purpose is clearer.
    - Instead of manually iterating through the list to calculate the sum, you can simplify the implementation by using Python’s built-in `sum()` function, which is both faster and more concise.

    Suggested Code:
    def calculate_sum(nums):
        try:
            return sum(nums)
        except TypeError:
            print("Error: Input must be iterable of numbers")
            return 0
    Case 2:
    - def multiply(a, b):
    -     return a * b
    + def multiply_numbers(x, y):
    +     return x * y
    - Deficiency Existence:
      - No

    Code Review Suggestion:
    - None

    Suggested Code:
    - None
    """

    prompt = prompt.strip("\n")
    payload = {
        "model": "deepseek-r1:14b",
        "messages": [
            { "role": "user", "content": prompt, "format": "json"}
        ],
        "stream": False,
        # "prompt": prompt,  # Include prompt if needed
        # "template": template_answer,  # Include template_answer if needed
        "keep_alive": "5m"
    }

    try:
      response = requests.post(endpoint, json=payload) # , auth=self.auth
      response.raise_for_status()
      return response.json()
    except requests.exceptions.RequestException as e:
      print(f"Error during request: {e}")
      return None

# Example usage:
# client = DeepSeekClient("127.0.0.1", 8080, "username", "password")
# result = client.perform_code_review("Review this code", "Expected output", "Bug detection")
# print(result)
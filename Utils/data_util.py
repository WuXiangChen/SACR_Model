import json

class JSONLReader:

  def __init__(self, file_path):
    self.file_path = file_path

  def read_lines(self, start=None, end=-1):
    """
    Read lines from the JSONL file and return them as a list of dictionaries.

    :param limit: Number of dictionaries to read. If None, read all lines.
    :return: List of dictionaries, each representing a JSON object from the file.
    """
    data = None
    try:
      with open(self.file_path, 'r', encoding='utf-8') as file:
          data = json.loads(file.readline().strip())
    except FileNotFoundError:
      print(f"Error: File not found at {self.file_path}")
    except json.JSONDecodeError as e:
      print(f"Error decoding JSON: {e}")
    if start is not None:
      return data[start:end]
    return data

  def write_lines(self, data):
    try:
      with open(self.file_path, 'w', encoding='utf-8') as file:
        for item in data:
          file.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
      print(f"Error writing to file: {e}")

  def filter_lines(self, condition):
    """
    Filter lines in the JSONL file based on a condition.

    :param condition: A function that takes a dictionary and returns True if it should be included.
    :return: List of dictionaries that satisfy the condition.
    """
    data = self.read_lines()
    return [item for item in data if condition(item)]

  def count_lines(self):
    """
    Count the number of lines (JSON objects) in the JSONL file.

    :return: Number of lines in the file.
    """
    return len(self.read_lines())

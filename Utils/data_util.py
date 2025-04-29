import json
import json
import os
import queue

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
  
def save_results(results_queue, model_name, dataset_name, output_dir="Results"):
    """
    Save results from the queue to a JSON file incrementally.
    Uses a consistent filename and appends new results to the existing file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate consistent filename (without timestamp)
    filename = f"{model_name}_{dataset_name}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Collect all results from the queue
    results = []
    while True:
        try:
            item = results_queue.get_nowait()
            if item is None:  # Our signal to stop
                break
            results.append(item)
        except queue.Empty:
            break
    
    # Append to JSON file if we have results
    if results:
        existing_data = []
        # Read existing data if file exists
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read existing file ({e}), starting fresh")
        
        # Combine old and new data
        combined_data = existing_data + results
        
        # Write back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nAdded {len(results)} results to {filepath} (now {len(combined_data)} total)")
    return len(results)

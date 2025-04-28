
server_address = "210.28.133.13"
port = 22207

username = "wxc@nju.edu.cn"
password = "wxc123"


DEEPSEEK_API_KEY = "sk-8c0a0b88ef384a08a290e1ee46fa8e82"
MONICA_API_KEY = "sk-TSlQVQ_di0Ns-h1j42zJvBH7HHZb3RF86T4n7hz67zoZ8iqgPFTM79cE7h1BcrBUgsPKMj_NOQKOvQHV9Cx_X2fuZMyu"

DS_BASE_URL = "https://api.deepseek.com"
MONICA_BASE_URL = "https://openapi.monica.im/v1"

## 这里需要实现一个字典，根据base_model来自动选择对应的URL和必要的参数
BASE_MODEL = {
  "ds_671B":{
    "API_KEY":DEEPSEEK_API_KEY,
    "BASE_URL":DS_BASE_URL,
    "MODEL_NAME":"deepseek-chat"
  },
  "gpt_4o":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"gpt-4o"
  },
  "ds_coder_v2":{
    "API_KEY":DEEPSEEK_API_KEY,
    "BASE_URL":DS_BASE_URL,
    "MODEL_NAME":"deepseek-coder-v2"
  },
  "ds_reasoner":{
    "API_KEY":DEEPSEEK_API_KEY,
    "BASE_URL":DS_BASE_URL,
    "MODEL_NAME":"deepseek-reasoner"
  },
  "qwen_2.5":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"qwen-2.5-72b-instruct"
  },
  "llama_3.1":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"llama-3.1-405b-instruct"
  }
}
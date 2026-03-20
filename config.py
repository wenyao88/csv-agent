import os

API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-uvlnnajaajjyqfnsnsqaxqkmlxskiitcovgevvkvkdvxdqem")
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "Qwen/Qwen2.5-32B-Instruct"
MAX_ITERATIONS = 5
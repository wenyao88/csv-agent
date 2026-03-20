import json
import pandas as pd
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL
from agents.base import AgentResult, make_success, make_error
from tools.preprocess_tools import (
    detect_data_info, minmax_normalize, zscore_normalize,
    fill_missing, drift_compensation, PREPROCESS_TOOLS
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一个电子鼻数据预处理专家。
你的任务是对传感器数据进行预处理，确保数据质量。

执行顺序：
1. 先调用detect_data_info了解数据情况
2. 如有缺失值，调用fill_missing处理
3. 根据任务需求选择归一化方式：
   - 需要保留原始分布用zscore_normalize
   - 需要统一量纲用minmax_normalize
4. 如果是时序传感器数据，调用drift_compensation做漂移补偿

每步工具调用后检查结果再决定下一步，处理完成后用中文简要总结做了哪些处理。"""


def run_preprocessor(df: pd.DataFrame, task: str) -> AgentResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"任务：{task}\n\n数据列名：{df.columns.tolist()}\n数据维度：{df.shape}"}
    ]

    current_df = df.copy()

    try:
        for _ in range(6):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=PREPROCESS_TOOLS,
                tool_choice="auto"
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return make_success(
                    output=msg.content,
                    data={"df": current_df, "original_df": df}
                )

            messages.append(msg)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                try:
                    if name == "detect_data_info":
                        result = detect_data_info(current_df)
                        content = str(result)

                    elif name == "minmax_normalize":
                        cols = args.get("cols", [])
                        valid_cols = [c for c in cols if c in current_df.columns]
                        current_df = minmax_normalize(current_df, valid_cols)
                        content = f"已对 {valid_cols} 完成Min-Max归一化"

                    elif name == "zscore_normalize":
                        cols = args.get("cols", [])
                        valid_cols = [c for c in cols if c in current_df.columns]
                        current_df = zscore_normalize(current_df, valid_cols)
                        content = f"已对 {valid_cols} 完成Z-score标准化"

                    elif name == "fill_missing":
                        cols = args.get("cols", [])
                        valid_cols = [c for c in cols if c in current_df.columns]
                        current_df, filled = fill_missing(current_df, valid_cols)
                        content = f"已填充 {filled} 个缺失值"

                    elif name == "drift_compensation":
                        cols = args.get("cols", [])
                        valid_cols = [c for c in cols if c in current_df.columns]
                        current_df = drift_compensation(current_df, valid_cols)
                        content = f"已对 {valid_cols} 完成漂移补偿"

                    else:
                        content = f"未知工具：{name}"

                except Exception as e:
                    content = f"工具执行出错：{str(e)}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })

        return make_success(output="预处理完成", data={"df": current_df, "original_df": df})

    except Exception as e:
        return make_error(f"预处理Agent异常：{str(e)}")
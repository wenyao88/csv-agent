import json
import pandas as pd
import io
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL, MAX_ITERATIONS
from tools import execute_python, TOOLS_DEFINITION

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def build_system_prompt(df: pd.DataFrame) -> str:
    sio = io.StringIO()
    df.info(buf=sio)
    return f"""你是一个专业的数据分析助手。用户上传了CSV文件，已加载为DataFrame变量df。

数据基本信息：
{sio.getvalue()}

前5行数据：
{df.head().to_string()}

规则：
1. 必须调用execute_python工具来分析数据，禁止凭空猜测任何数值
2. 用户要求画图时，必须在代码中用plotly生成图表并赋值给变量fig
3. 禁止在回复文字中使用markdown图片语法或描述图表占位符
4. 代码执行出错时，仔细分析错误原因，修正后重新调用工具
5. 最终用简洁中文给出结论，图表会自动渲染无需描述"""


def extract_code(arguments) -> str:
    if isinstance(arguments, dict):
        return arguments.get("code", "")
    try:
        parsed = json.loads(str(arguments))
        return parsed.get("code", "") if isinstance(parsed, dict) else str(parsed)
    except Exception:
        return str(arguments)


def clean_response(text: str) -> str:
    import re
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|.*?\|>', '', text)
    return text.strip()


def run_agent(user_message: str, df: pd.DataFrame, history: list) -> tuple[str, any, list]:
    messages = [{"role": "system", "content": build_system_prompt(df)}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    fig = None
    steps = []

    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DEFINITION,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                code = extract_code(tool_call.function.arguments)
                output, returned_fig = execute_python(code, df)

                step = {
                    "iteration": i + 1,
                    "code": code,
                    "output": output,
                    "has_fig": returned_fig is not None
                }
                steps.append(step)

                if returned_fig is not None:
                    fig = returned_fig

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output
                })
        else:
            return clean_response(msg.content), fig, steps

    return "已达到最大分析轮次，请尝试更具体的问题。", fig, steps
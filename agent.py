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
    return f"""你是一个数据分析助手。用户上传了一个CSV文件，已加载为DataFrame变量df。

数据基本信息：
{sio.getvalue()}

前5行数据：
{df.head().to_string()}

规则：
1. 必须调用execute_python工具来分析数据，禁止凭空猜测数值
2. 用户要求画图时，必须调用execute_python工具，在代码中用plotly生成图表并赋值给变量fig，禁止在文字回复中用markdown语法描述图表
3. 分析完成后用简洁的中文总结结论，不要描述图表内容，图表会自动显示
4. 如果代码执行出错，分析原因并修正后重试"""


def extract_code(arguments) -> str:
    if isinstance(arguments, dict):
        return arguments.get("code", "")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed.get("code", "")
            return str(parsed)
        except Exception:
            return arguments
    try:
        d = dict(arguments)
        return d.get("code", str(arguments))
    except Exception:
        return str(arguments)


def run_agent(user_message: str, df: pd.DataFrame, history: list) -> tuple[str, any]:
    messages = [{"role": "system", "content": build_system_prompt(df)}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    fig = None

    for _ in range(MAX_ITERATIONS):
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
                if returned_fig is not None:
                    fig = returned_fig
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output
                })
        else:
            return msg.content, fig

    return "已达到最大分析轮次，请尝试更具体的问题。", fig
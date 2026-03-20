import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import sys


def execute_python(code: str, df: pd.DataFrame) -> tuple[str, any]:
    local_env = {
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "df": df,
        "fig": None,
    }
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    try:
        exec(code, local_env)
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"代码执行出错：{str(e)}", None
    finally:
        sys.stdout = sys.__stdout__

    output = stdout_capture.getvalue().strip()
    fig = local_env.get("fig", None)

    if not output and fig is None:
        try:
            last_line = code.strip().split("\n")[-1]
            result = eval(last_line, local_env)
            output = str(result)
        except:
            output = "代码执行完成，无文字输出"

    return output or "代码执行完成", fig


TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "执行Python代码分析DataFrame数据。数据已加载为变量df。可使用pd/np/px/go。如需生成图表，将plotly图表赋值给变量fig。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的Python代码"
                    }
                },
                "required": ["code"]
            }
        }
    }
]
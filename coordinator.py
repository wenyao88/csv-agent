import re
import pandas as pd
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL
from agents.preprocessor import run_preprocessor
from agents.analyzer import run_analyzer
from agents.visualizer import run_visualizer
from agents.explainer import run_explainer
from agents.base import AgentResult

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

INTENT_PROMPT = """你是一个任务分析器。根据用户问题判断需要执行哪些分析步骤。

可选步骤：
- preprocess：数据预处理（归一化、缺失值、漂移补偿）
- analyze：统计分析（PCA、聚类、区分能力）
- visualize：生成图表（雷达图、PCA图、热力图等）
- explain：化学专业解释

规则：
- 只问数据基本情况 → ["preprocess"]
- 要求分析/统计 → ["preprocess", "analyze"]
- 要求画图/可视化 → ["preprocess", "analyze", "visualize"]
- 要求综合分析/解释/报告 → ["preprocess", "analyze", "visualize", "explain"]
- 要求化学解释 → ["preprocess", "analyze", "explain"]

只返回JSON数组，不要其他内容。例如：["preprocess", "analyze", "visualize"]"""


def detect_label_col(df: pd.DataFrame) -> str | None:
    candidates = ["label", "gas", "class", "type", "category",
                  "标签", "气体", "类别", "种类"]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() <= 20:
            return col
    return None


def parse_intent(user_message: str) -> list:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=50
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
        import json
        steps = json.loads(raw)
        valid = {"preprocess", "analyze", "visualize", "explain"}
        return [s for s in steps if s in valid]
    except Exception:
        return ["preprocess", "analyze", "visualize"]


def run_with_retry(fn, *args, max_retry=2, **kwargs) -> AgentResult:
    for attempt in range(max_retry):
        try:
            result = fn(*args, **kwargs)
            if result.success:
                return result
            if attempt < max_retry - 1:
                continue
        except Exception as e:
            if attempt == max_retry - 1:
                from agents.base import make_error
                return make_error(str(e))
    from agents.base import make_error
    return make_error("多次重试后仍然失败")


def run_coordinator(user_message: str, df: pd.DataFrame,
                    history: list) -> tuple[str, list, list]:
    steps = parse_intent(user_message)
    label_col = detect_label_col(df)

    figs = []
    step_logs = []
    preprocess_output = ""
    analysis_output = ""
    analysis_data = {}

    if "preprocess" in steps:
        step_logs.append({"agent": "预处理Agent", "status": "running"})
        result = run_with_retry(run_preprocessor, df, user_message)
        if result.success:
            df = result.data.get("df", df)
            preprocess_output = result.output
            step_logs[-1]["status"] = "success"
            step_logs[-1]["output"] = result.output
        else:
            step_logs[-1]["status"] = "error"
            step_logs[-1]["output"] = result.error_msg
            preprocess_output = "预处理失败，使用原始数据继续"

    if "analyze" in steps:
        step_logs.append({"agent": "分析Agent", "status": "running"})
        result = run_with_retry(run_analyzer, df, user_message, label_col)
        if result.success:
            analysis_output = result.output
            analysis_data = result.data
            step_logs[-1]["status"] = "success"
            step_logs[-1]["output"] = result.output
        else:
            step_logs[-1]["status"] = "error"
            step_logs[-1]["output"] = result.error_msg
            analysis_output = "分析失败"
            analysis_data = {"df": df, "sensor_cols": df.select_dtypes(
                include="number").columns.tolist(), "label_col": label_col,
                             "analysis_results": {}}

    if "visualize" in steps:
        step_logs.append({"agent": "可视化Agent", "status": "running"})
        result = run_with_retry(run_visualizer, analysis_data, user_message)
        if result.success:
            figs = result.figs
            step_logs[-1]["status"] = "success"
            step_logs[-1]["output"] = result.output
        else:
            step_logs[-1]["status"] = "error"
            step_logs[-1]["output"] = result.error_msg

    final_answer = ""
    if "explain" in steps:
        step_logs.append({"agent": "化学解释Agent", "status": "running"})
        result = run_with_retry(
            run_explainer, user_message,
            preprocess_output, analysis_output, analysis_data
        )
        if result.success:
            final_answer = result.output
            step_logs[-1]["status"] = "success"
            step_logs[-1]["output"] = result.output
        else:
            step_logs[-1]["status"] = "error"
            step_logs[-1]["output"] = result.error_msg

    if not final_answer:
        parts = []
        if preprocess_output:
            parts.append(f"**预处理：** {preprocess_output}")
        if analysis_output:
            parts.append(f"**分析结果：** {analysis_output}")
        if figs:
            parts.append(f"**可视化：** 已生成 {len(figs)} 张图表")
        final_answer = "\n\n".join(parts) if parts else "分析完成"

    return final_answer, figs, step_logs
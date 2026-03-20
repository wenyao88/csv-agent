import json
import pandas as pd
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL
from agents.base import AgentResult, make_success, make_error
from tools.analysis_tools import (
    pca_analysis, kmeans_cluster,
    sensor_response_stats, discrimination_power,
    ANALYSIS_TOOLS
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一个电子鼻数据分析专家，擅长传感器阵列数据的模式识别与统计分析。

执行原则：
1. 先调用sensor_response_stats了解各传感器的响应特征
2. 如果数据有气体标签列，调用discrimination_power找出区分能力最强的传感器
3. 调用pca_analysis做降维，观察气体样本的分布规律
4. 如果需要发现数据中的分组规律，调用kmeans_cluster做聚类
5. 每步结果作为下一步的依据，分析完成后给出中文总结"""


def run_analyzer(df: pd.DataFrame, task: str, label_col: str = None) -> AgentResult:
    sensor_cols = df.select_dtypes(include="number").columns.tolist()
    if label_col and label_col in sensor_cols:
        sensor_cols.remove(label_col)

    context = f"""任务：{task}
数据维度：{df.shape}
传感器列：{sensor_cols}
标签列：{label_col if label_col else '无'}
前3行数据：
{df.head(3).to_string()}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context}
    ]

    analysis_results = {}

    try:
        for _ in range(8):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=ANALYSIS_TOOLS,
                tool_choice="auto"
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return make_success(
                    output=msg.content,
                    data={
                        "analysis_results": analysis_results,
                        "sensor_cols": sensor_cols,
                        "label_col": label_col,
                        "df": df
                    }
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
                    if name == "pca_analysis":
                        cols = [c for c in args.get("cols", sensor_cols) if c in df.columns]
                        n = args.get("n_components", 2)
                        result = pca_analysis(df, cols, n)
                        analysis_results["pca"] = result
                        content = (f"PCA完成：{result['n_components']}个主成分，"
                                   f"累计方差贡献率 {result['explained_variance_total']*100:.1f}%，"
                                   f"各主成分贡献：{[f'{v*100:.1f}%' for v in result['explained_variance']]}")

                    elif name == "kmeans_cluster":
                        cols = [c for c in args.get("cols", sensor_cols) if c in df.columns]
                        k = args.get("n_clusters", 3)
                        result = kmeans_cluster(df, cols, k)
                        analysis_results["kmeans"] = result
                        content = (f"K-Means聚类完成：{k}个簇，"
                                   f"惯性值 {result['inertia']}，"
                                   f"各簇样本数：{result['cluster_counts']}")

                    elif name == "sensor_response_stats":
                        cols = [c for c in args.get("cols", sensor_cols) if c in df.columns]
                        lbl = args.get("label_col", label_col)
                        result = sensor_response_stats(df, cols, lbl)
                        analysis_results["stats"] = result
                        content = f"传感器统计完成，共分析 {len(cols)} 个传感器"
                        if "group_means" in result:
                            content += f"，气体类别：{result['labels']}"

                    elif name == "discrimination_power":
                        cols = [c for c in args.get("cols", sensor_cols) if c in df.columns]
                        lbl = args.get("label_col", label_col)
                        if not lbl:
                            content = "未提供标签列，跳过区分能力分析"
                        else:
                            result = discrimination_power(df, cols, lbl)
                            analysis_results["discrimination"] = result
                            content = (f"区分能力分析完成，"
                                       f"最优传感器：{result['best_sensors']}")
                    else:
                        content = f"未知工具：{name}"

                except Exception as e:
                    content = f"工具执行出错：{str(e)}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })

        return make_success(
            output="分析完成",
            data={
                "analysis_results": analysis_results,
                "sensor_cols": sensor_cols,
                "label_col": label_col,
                "df": df
            }
        )

    except Exception as e:
        return make_error(f"分析Agent异常：{str(e)}")
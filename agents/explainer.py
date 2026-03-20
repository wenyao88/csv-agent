import pandas as pd
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL
from agents.base import AgentResult, make_success, make_error

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一个电子鼻与化学传感器领域的专家，擅长将数据分析结果转化为专业的化学解释。

你的任务是根据提供的分析结果，给出专业、准确、易懂的化学层面解释。

解释要包含：
1. 传感器阵列对气体的整体识别能力评估
2. PCA结果的化学含义（气体可分性、传感器选择性）
3. 聚类结果与气体种类的对应关系
4. 区分能力最强的传感器可能对哪类化合物敏感
5. 数据质量评估与实际应用建议

语言要求：专业但易懂，避免纯数字堆砌，要给出化学意义上的解释。"""


def build_context(task: str, preprocess_output: str,
                  analysis_output: str, analysis_data: dict) -> str:
    results = analysis_data.get("analysis_results", {})
    sensor_cols = analysis_data.get("sensor_cols", [])
    label_col = analysis_data.get("label_col")

    parts = [f"用户任务：{task}\n"]
    parts.append(f"传感器数量：{len(sensor_cols)}")
    parts.append(f"传感器列表：{sensor_cols[:16]}")
    if label_col:
        parts.append(f"气体标签列：{label_col}")

    parts.append(f"\n预处理结果：\n{preprocess_output}")
    parts.append(f"\n分析结果摘要：\n{analysis_output}")

    if "pca" in results:
        pca = results["pca"]
        parts.append(f"\nPCA详情：")
        parts.append(f"  主成分数：{pca['n_components']}")
        parts.append(f"  各主成分方差贡献率：{[f'{v*100:.1f}%' for v in pca['explained_variance']]}")
        parts.append(f"  累计贡献率：{pca['explained_variance_total']*100:.1f}%")

    if "kmeans" in results:
        km = results["kmeans"]
        parts.append(f"\n聚类详情：")
        parts.append(f"  聚类数：{km['n_clusters']}")
        parts.append(f"  各簇样本数：{km['cluster_counts']}")
        parts.append(f"  聚类惯性值：{km['inertia']}")

    if "discrimination" in results:
        disc = results["discrimination"]
        parts.append(f"\n传感器区分能力：")
        parts.append(f"  最优传感器（前3）：{disc['best_sensors']}")
        top5 = list(disc["discrimination_scores"].items())[:5]
        parts.append(f"  区分得分前5：{top5}")

    if "stats" in results and "labels" in results["stats"]:
        parts.append(f"\n数据包含气体种类：{results['stats']['labels']}")

    return "\n".join(parts)


def run_explainer(task: str, preprocess_output: str,
                  analysis_output: str, analysis_data: dict) -> AgentResult:
    try:
        context = build_context(task, preprocess_output, analysis_output, analysis_data)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        explanation = response.choices[0].message.content
        return make_success(output=explanation)

    except Exception as e:
        return make_error(f"化学解释Agent异常：{str(e)}")
import pandas as pd
from agents.base import AgentResult, make_success, make_error
from tools.viz_tools import (
    radar_chart, pca_scatter, sensor_heatmap,
    response_curves, discrimination_bar
)


def run_visualizer(analysis_data: dict, task: str) -> AgentResult:
    try:
        df = analysis_data.get("df")
        sensor_cols = analysis_data.get("sensor_cols", [])
        label_col = analysis_data.get("label_col")
        results = analysis_data.get("analysis_results", {})

        if df is None or len(sensor_cols) == 0:
            return make_error("缺少数据或传感器列信息")

        figs = []
        descriptions = []

        task_lower = task.lower()
        keywords = {
            "雷达": ["雷达", "radar", "响应模式", "传感器模式"],
            "pca": ["pca", "主成分", "降维", "散点"],
            "热力": ["热力", "heatmap", "相关", "热图"],
            "曲线": ["曲线", "时序", "响应曲线", "变化"],
            "区分": ["区分", "discrimination", "排名", "重要性"]
        }

        requested = set()
        for key, words in keywords.items():
            if any(w in task_lower for w in words):
                requested.add(key)

        if not requested:
            requested = {"雷达", "pca", "热力"}

        if "雷达" in requested and len(sensor_cols) >= 3:
            try:
                fig = radar_chart(df, sensor_cols[:12], label_col)
                figs.append(fig)
                descriptions.append("传感器响应雷达图：展示各传感器对不同气体的响应强度模式")
            except Exception as e:
                descriptions.append(f"雷达图生成失败：{e}")

        if "pca" in requested and "pca" in results:
            try:
                pca_result = results["pca"]
                labels = None
                if label_col and label_col in df.columns:
                    labels = df[label_col].values[:len(pca_result["pca_data"])]
                fig = pca_scatter(pca_result["pca_data"], labels, label_col or "类别")
                figs.append(fig)
                var_total = pca_result["explained_variance_total"] * 100
                descriptions.append(f"PCA散点图：前两个主成分累计解释 {var_total:.1f}% 的方差")
            except Exception as e:
                descriptions.append(f"PCA图生成失败：{e}")

        if "热力" in requested:
            try:
                fig = sensor_heatmap(df, sensor_cols[:16], label_col)
                figs.append(fig)
                descriptions.append("热力图：展示传感器响应强度分布")
            except Exception as e:
                descriptions.append(f"热力图生成失败：{e}")

        if "曲线" in requested:
            try:
                fig = response_curves(df, sensor_cols, label_col)
                figs.append(fig)
                descriptions.append("传感器响应曲线：各传感器随样本变化的响应趋势")
            except Exception as e:
                descriptions.append(f"响应曲线生成失败：{e}")

        if "区分" in requested and "discrimination" in results:
            try:
                scores = results["discrimination"]["discrimination_scores"]
                fig = discrimination_bar(scores)
                figs.append(fig)
                descriptions.append("区分能力排名：得分越高的传感器对气体区分越有效")
            except Exception as e:
                descriptions.append(f"区分能力图生成失败：{e}")

        if not figs:
            return make_error("没有生成任何图表，请检查数据或分析结果")

        output = f"共生成 {len(figs)} 张图表：\n" + "\n".join(f"- {d}" for d in descriptions)
        return make_success(output=output, figs=figs)

    except Exception as e:
        return make_error(f"可视化Agent异常：{str(e)}")
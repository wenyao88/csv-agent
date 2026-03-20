import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def pca_analysis(df: pd.DataFrame, cols: list, n_components: int = 2) -> dict:
    X = df[cols].dropna().values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    return {
        "pca_data": pd.DataFrame(
            X_pca,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        ),
        "explained_variance": [round(float(v), 4) for v in explained],
        "explained_variance_total": round(float(sum(explained)), 4),
        "loadings": loadings,
        "n_components": pca.n_components_
    }


def kmeans_cluster(df: pd.DataFrame, cols: list, n_clusters: int = 3) -> dict:
    X = df[cols].dropna().values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia = float(km.inertia_)
    cluster_counts = pd.Series(labels).value_counts().sort_index().to_dict()
    return {
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "inertia": round(inertia, 4),
        "cluster_counts": {f"簇{k}": v for k, v in cluster_counts.items()}
    }


def sensor_response_stats(df: pd.DataFrame, cols: list, label_col: str = None) -> dict:
    stats = df[cols].describe().round(4)
    result = {"overall_stats": stats.to_dict()}
    if label_col and label_col in df.columns:
        group_means = df.groupby(label_col)[cols].mean().round(4)
        result["group_means"] = group_means.to_dict()
        result["labels"] = df[label_col].unique().tolist()
    return result


def discrimination_power(df: pd.DataFrame, cols: list, label_col: str) -> dict:
    if label_col not in df.columns:
        return {"error": f"标签列 {label_col} 不存在"}
    result = {}
    for col in cols:
        groups = [df[df[label_col] == lbl][col].dropna().values
                  for lbl in df[label_col].unique()]
        overall_mean = df[col].mean()
        overall_var = df[col].var()
        if overall_var > 0:
            between_var = np.var([g.mean() for g in groups if len(g) > 0])
            score = float(between_var / overall_var)
        else:
            score = 0.0
        result[col] = round(score, 4)
    ranked = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    return {"discrimination_scores": ranked,
            "best_sensors": list(ranked.keys())[:3]}


ANALYSIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pca_analysis",
            "description": "对传感器数据做PCA主成分分析，返回降维结果、方差贡献率和载荷矩阵",
            "parameters": {
                "type": "object",
                "properties": {
                    "cols": {"type": "array", "items": {"type": "string"}, "description": "参与PCA的列名"},
                    "n_components": {"type": "integer", "description": "主成分数量，默认2", "default": 2}
                },
                "required": ["cols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kmeans_cluster",
            "description": "对传感器数据做K-Means聚类分析",
            "parameters": {
                "type": "object",
                "properties": {
                    "cols": {"type": "array", "items": {"type": "string"}, "description": "参与聚类的列名"},
                    "n_clusters": {"type": "integer", "description": "聚类数量，默认3", "default": 3}
                },
                "required": ["cols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sensor_response_stats",
            "description": "计算各传感器响应值的统计特征，可按气体类别分组",
            "parameters": {
                "type": "object",
                "properties": {
                    "cols": {"type": "array", "items": {"type": "string"}, "description": "传感器列名"},
                    "label_col": {"type": "string", "description": "气体类别标签列名，可选"}
                },
                "required": ["cols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "discrimination_power",
            "description": "计算每个传感器对不同气体的区分能力得分，得分越高说明该传感器对气体区分越有效",
            "parameters": {
                "type": "object",
                "properties": {
                    "cols": {"type": "array", "items": {"type": "string"}, "description": "传感器列名"},
                    "label_col": {"type": "string", "description": "气体类别标签列名"}
                },
                "required": ["cols", "label_col"]
            }
        }
    }
]
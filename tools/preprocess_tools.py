import pandas as pd
import numpy as np


def detect_data_info(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    missing = df.isnull().sum().sum()
    return {
        "shape": df.shape,
        "numeric_cols": numeric_cols,
        "missing_count": int(missing),
        "columns": df.columns.tolist()
    }


def minmax_normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    result = df.copy()
    for col in cols:
        min_val = result[col].min()
        max_val = result[col].max()
        if max_val - min_val > 0:
            result[col] = (result[col] - min_val) / (max_val - min_val)
        else:
            result[col] = 0.0
    return result


def zscore_normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    result = df.copy()
    for col in cols:
        mean = result[col].mean()
        std = result[col].std()
        if std > 0:
            result[col] = (result[col] - mean) / std
        else:
            result[col] = 0.0
    return result


def fill_missing(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, int]:
    result = df.copy()
    filled = 0
    for col in cols:
        n = result[col].isnull().sum()
        if n > 0:
            result[col] = result[col].fillna(result[col].median())
            filled += n
    return result, filled


def drift_compensation(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    result = df.copy()
    for col in cols:
        baseline = result[col].iloc[:max(1, len(result) // 10)].mean()
        if baseline != 0:
            result[col] = result[col] / baseline
    return result


PREPROCESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_data_info",
            "description": "检测数据基本信息，返回数值列名、缺失值数量、数据维度",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "minmax_normalize",
            "description": "对指定列做Min-Max归一化，将数值压缩到[0,1]区间",
            "parameters": {
                "type": "object",
                "properties": {
                    "cols": {"type": "array", "items": {"type": "string"}, "description": "要归一化的列名列表"}
                },
                "required": ["cols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "zscore_normalize",
            "description": "对指定列做Z-score标准化",
            "parameters": {
                "type": "object",
                "properties": {
                    "cols": {"type": "array", "items": {"type": "string"}, "description": "要标准化的列名列表"}
                },
                "required": ["cols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fill_missing",
            "description": "用中位数填充指定列的缺失值",
            "parameters": {
                "type": "object",
                "properties": {
                    "cols": {"type": "array", "items": {"type": "string"}, "description": "要填充的列名列表"}
                },
                "required": ["cols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drift_compensation",
            "description": "对传感器数据做基线漂移补偿，消除传感器老化导致的系统性偏移",
            "parameters": {
                "type": "object",
                "properties": {
                    "cols": {"type": "array", "items": {"type": "string"}, "description": "要做漂移补偿的传感器列名列表"}
                },
                "required": ["cols"]
            }
        }
    }
]
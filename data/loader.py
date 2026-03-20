import pandas as pd
import numpy as np


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"不支持的文件格式：{name}")


def generate_demo_data(n_samples: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    gases = ["乙醇", "乙烯", "氨气", "丙酮", "甲醛", "乙酸乙酯"]
    n_per_gas = n_samples // len(gases)

    base_responses = {
        "乙醇":    [0.8, 0.6, 0.3, 0.2, 0.1, 0.1, 0.4, 0.5, 0.7, 0.6, 0.2, 0.3, 0.5, 0.4, 0.3, 0.2],
        "乙烯":    [0.2, 0.3, 0.7, 0.8, 0.6, 0.5, 0.2, 0.1, 0.3, 0.2, 0.6, 0.7, 0.4, 0.5, 0.6, 0.7],
        "氨气":    [0.1, 0.2, 0.2, 0.3, 0.8, 0.9, 0.7, 0.6, 0.2, 0.1, 0.3, 0.2, 0.7, 0.8, 0.9, 0.8],
        "丙酮":    [0.5, 0.6, 0.4, 0.3, 0.2, 0.1, 0.8, 0.9, 0.5, 0.6, 0.4, 0.3, 0.2, 0.1, 0.3, 0.4],
        "甲醛":    [0.3, 0.2, 0.5, 0.6, 0.3, 0.4, 0.2, 0.3, 0.8, 0.9, 0.7, 0.6, 0.3, 0.2, 0.4, 0.5],
        "乙酸乙酯": [0.6, 0.7, 0.2, 0.1, 0.4, 0.3, 0.5, 0.4, 0.4, 0.3, 0.8, 0.9, 0.6, 0.7, 0.5, 0.6],
    }

    rows = []
    for gas, base in base_responses.items():
        for i in range(n_per_gas):
            drift = 1 + 0.002 * i
            noise = np.random.normal(0, 0.03, 16)
            row = [max(0, b * drift + n) for b, n in zip(base, noise)]
            rows.append(row + [gas])

    sensor_cols = [f"S{i+1}" for i in range(16)]
    df = pd.DataFrame(rows, columns=sensor_cols + ["gas"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)
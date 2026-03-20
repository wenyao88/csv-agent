# 电子鼻传感数据智能分析平台

基于 Multi-Agent 架构的电子鼻传感器阵列数据分析系统，支持自然语言交互，自动完成数据预处理、模式识别、可视化与化学专业解释。

## 项目演示

> 上传传感器数据或使用内置演示数据，用自然语言提问，系统自动调度多个专业 Agent 协作完成分析。

**示例问题：**
- 对数据做全面分析并给出化学解释
- 分析各传感器的气体区分能力
- 做 PCA 分析并画散点图
- 画传感器响应雷达图

## 核心架构
```
用户提问
    ↓
协调者 Agent（意图识别 → 任务路由 → 错误重试）
    ├→ 预处理 Agent：归一化 / 缺失值填充 / 漂移补偿
    ├→ 分析 Agent：PCA / K-Means 聚类 / 传感器区分能力
    ├→ 可视化 Agent：雷达图 / PCA 散点图 / 热力图
    └→ 化学解释 Agent：将数值结果转化为化学语言描述
```

### 关键设计

**标准化 AgentResult 接口**：每个 Agent 返回统一的 `status / output / data / figs / error_msg` 结构，协调者通过 `status` 字段感知每步成败，失败自动重试，重试仍失败则跳过并继续后续步骤，保证流程不中断。

**三层错误保障**：Python 异常捕获 + AgentResult status 检查 + 最多两次自动重试。

**动态任务路由**：协调者用 LLM 理解用户意图，动态决定调用哪些 Agent 的组合，而非硬编码 if-else。

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM | Qwen2.5-32B-Instruct（硅基流动 API）|
| Agent 框架 | 自实现 ReAct 模式 + Function Calling |
| 数据分析 | Pandas / NumPy / Scikit-learn |
| 可视化 | Plotly（雷达图 / PCA 双标图 / 热力图）|
| 前端 | Streamlit |
| 部署 | Cloudflare Tunnel |

## 本地运行
```bash
git clone https://github.com/wenyao88/enose-agent.git
cd enose-agent

pip install -r requirements.txt

cp config.example.py config.py
# 编辑 config.py，填入你的硅基流动 API Key

streamlit run app.py
```

## 数据格式

支持 CSV / Excel 文件，传感器列为数值型，可选包含气体标签列（列名含 label / gas / class / 气体 / 类别 等关键词时自动识别）。

内置演示数据：16 个金属氧化物传感器模拟数据，6 种气体（乙醇、乙烯、氨气、丙酮、甲醛、乙酸乙酯），200 条样本，含传感器漂移模拟。

真实数据集：[UCI Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset)（16 传感器 / 6 气体 / 13910 样本）

## 项目结构
```
enose-agent/
├── app.py                  # Streamlit 界面
├── coordinator.py          # 协调者 Agent
├── agents/
│   ├── base.py             # AgentResult 标准接口
│   ├── preprocessor.py     # 预处理 Agent
│   ├── analyzer.py         # 分析 Agent
│   ├── visualizer.py       # 可视化 Agent
│   └── explainer.py        # 化学解释 Agent
├── tools/
│   ├── preprocess_tools.py # 归一化 / 漂移补偿工具
│   ├── analysis_tools.py   # PCA / 聚类 / 统计工具
│   └── viz_tools.py        # 图表生成工具
├── data/
│   └── loader.py           # 数据加载 / 演示数据生成
└── config.example.py       # 配置模板
```
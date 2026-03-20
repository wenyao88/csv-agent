import streamlit as st
import pandas as pd
from coordinator import run_coordinator
from data.loader import load_uploaded_file, generate_demo_data

st.set_page_config(page_title="电子鼻智能分析平台", layout="wide")
st.title("电子鼻传感数据智能分析平台")
st.caption("基于 Multi-Agent 架构 · 支持 PCA / 聚类 / 传感器区分能力分析 · 化学专业解释")

if "history" not in st.session_state:
    st.session_state.history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "messages_display" not in st.session_state:
    st.session_state.messages_display = []
if "last_file" not in st.session_state:
    st.session_state.last_file = None


def show_data_profile(df: pd.DataFrame):
    st.subheader("数据概览")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("样本数", df.shape[0])
    col2.metric("传感器数", len(df.select_dtypes(include="number").columns))
    col3.metric("总列数", df.shape[1])
    col4.metric("缺失值", int(df.isnull().sum().sum()))

    with st.expander("字段详情"):
        info = pd.DataFrame({
            "字段": df.columns,
            "类型": df.dtypes.values,
            "缺失数": df.isnull().sum().values,
            "唯一值数": df.nunique().values
        })
        st.dataframe(info, use_container_width=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        with st.expander("传感器响应统计"):
            st.dataframe(df[num_cols].describe().round(4), use_container_width=True)


def show_step_logs(step_logs: list):
    status_icon = {"success": "✅", "error": "❌", "running": "⏳"}
    for log in step_logs:
        icon = status_icon.get(log["status"], "•")
        with st.expander(f"{icon} {log['agent']}"):
            if log.get("output"):
                st.markdown(log["output"])


with st.sidebar:
    st.header("数据加载")
    use_demo = st.toggle("使用内置演示数据", value=False)

    if use_demo:
        if st.session_state.last_file != "__demo__":
            st.session_state.df = generate_demo_data()
            st.session_state.last_file = "__demo__"
            st.session_state.history = []
            st.session_state.messages_display = []
        st.success("已加载演示数据：200样本 · 16传感器 · 6种气体")
    else:
        uploaded_file = st.file_uploader("上传传感器数据", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            if uploaded_file.name != st.session_state.last_file:
                try:
                    st.session_state.df = load_uploaded_file(uploaded_file)
                    st.session_state.last_file = uploaded_file.name
                    st.session_state.history = []
                    st.session_state.messages_display = []
                    st.success(f"已加载：{st.session_state.df.shape[0]}行 × {st.session_state.df.shape[1]}列")
                except Exception as e:
                    st.error(f"读取失败：{e}")

    st.divider()
    st.header("示例问题")
    example_questions = [
        "对数据做预处理并归一化",
        "分析各传感器的气体区分能力",
        "做PCA分析并画散点图",
        "画传感器响应雷达图",
        "对数据做全面分析并给出化学解释",
        "哪些传感器对气体识别最重要",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["preset_question"] = q

    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages_display = []
        st.rerun()

if st.session_state.df is None:
    st.info("请在左侧上传数据文件，或开启演示数据开关")
    st.stop()

show_data_profile(st.session_state.df)
st.divider()

for item in st.session_state.messages_display:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("step_logs"):
            with st.expander("🔍 查看 Agent 执行过程"):
                show_step_logs(item["step_logs"])
        for fig in item.get("figs", []):
            st.plotly_chart(fig, use_container_width=True)

prompt = st.chat_input("描述你想对电子鼻数据做什么分析...")
if "preset_question" in st.session_state:
    prompt = st.session_state.pop("preset_question")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("多Agent协作分析中..."):
            answer, figs, step_logs = run_coordinator(
                prompt, st.session_state.df.copy(), st.session_state.history
            )
        st.markdown(answer)
        with st.expander("🔍 查看 Agent 执行过程"):
            show_step_logs(step_logs)
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)

    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.messages_display.append({"role": "user", "content": prompt})
    st.session_state.messages_display.append({
        "role": "assistant", "content": answer,
        "step_logs": step_logs, "figs": figs
    })
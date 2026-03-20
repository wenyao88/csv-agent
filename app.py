import streamlit as st
import pandas as pd
from agent import run_agent

st.set_page_config(page_title="CSV智能分析助手", layout="wide")
st.title("CSV 智能分析助手")

if "history" not in st.session_state:
    st.session_state.history = []
if "df" not in st.session_state:
    st.session_state.df = None

with st.sidebar:
    st.header("上传数据")
    uploaded_file = st.file_uploader("选择CSV文件", type=["csv"])
    if uploaded_file:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"已加载：{st.session_state.df.shape[0]}行 × {st.session_state.df.shape[1]}列")
            st.dataframe(st.session_state.df.head())
        except Exception as e:
            st.error(f"读取失败：{e}")
    if st.button("清空对话"):
        st.session_state.history = []
        st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.df is None:
    st.info("请先在左侧上传CSV文件")
else:
    if prompt := st.chat_input("问关于数据的任何问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                answer, fig = run_agent(prompt, st.session_state.df, st.session_state.history)
            st.markdown(answer)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": answer})
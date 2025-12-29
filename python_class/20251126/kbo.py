import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np


st.set_page_config(page_title="KBO 선수들의 연봉 히스토그램")


st.sidebar.header("팀 선택하기")


kbo = pd.read_pickle("./kbo_salary.pkl")


with st.sidebar:
    selected = st.selectbox('팀 선택', kbo.team.unique())


data = kbo.loc[kbo['team'] == selected, '연봉'].values


fig = px.histogram(data, nbins=30)


st.plotly_chart(fig, use_container_width=True)
st.write(kbo[kbo['team'] == selected])



import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

st.title("🆕 Acompanhamento de Novos")

# =========================
# Arquivo
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARQ_NOVOS = DATA_DIR / "Acomp novos.xlsx"

@st.cache_data(show_spinner=True)
def carregar_novos():
    if not ARQ_NOVOS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_NOVOS}")
    return pd.read_excel(ARQ_NOVOS)

try:
    df = carregar_novos()
except Exception as e:
    st.error(f"Erro ao carregar arquivo: {e}")
    st.stop()

# =========================
# Normalizar colunas
# =========================
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.upper()
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)

# =========================
# Datas e tempo de casa
# =========================
if "ADMISSAO" in df.columns:
    df["ADMISSAO"] = pd.to_datetime(df["ADMISSAO"], errors="coerce", dayfirst=True)

hoje = pd.to_datetime(datetime.today().date())

if "TEMPO DE CASA" not in df.columns or df["TEMPO DE CASA"].isna().all():
    df["TEMPO DE CASA"] = (hoje - df["ADMISSAO"]).dt.days

df["TEMPO DE CASA"] = df["TEMPO DE CASA"].fillna(0).astype(int)

# =========================
# Sidebar filtros
# =========================
st.sidebar.header("Filtros")

def opcoes(col):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique())

f_regional = st.sidebar.multiselect("Regional", opcoes("REGIONAL"))
f_empresa = st.sidebar.multiselect("Empresa", opcoes("EMPRESA"))
f_operacao = st.sidebar.multiselect("Operação", opcoes("OPERACAO"))
f_atividade = st.sidebar.multiselect("Atividade", opcoes("ATIVIDADE"))

df_f = df.copy()

if f_regional:
    df_f = df_f[df_f["REGIONAL"].isin(f_regional)]
if f_empresa:
    df_f = df_f[df_f["EMPRESA"].isin(f_empresa)]
if f_operacao:
    df_f = df_f[df_f["OPERACAO"].isin(f_operacao)]
if f_atividade:
    df_f = df_f[df_f["ATIVIDADE"].isin(f_atividade)]

# =========================
# Cards
# =========================
total = len(df_f)
ate30 = len(df_f[df_f["TEMPO DE CASA"] <= 30])
ate60 = len(df_f[(df_f["TEMPO DE CASA"] > 30) & (df_f["TEMPO DE CASA"] <= 60)])
ate90 = len(df_f[(df_f["TEMPO DE CASA"] > 60) & (df_f["TEMPO DE CASA"] <= 90)])
mais90 = len(df_f[df_f["TEMPO DE CASA"] > 90])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total", total)
c2.metric("Até 30 dias", ate30)
c3.metric("31–60 dias", ate60)
c4.metric("61–90 dias", ate90)
c5.metric("+90 dias", mais90)

st.divider()

# =========================
# Tabela
# =========================
st.subheader("📋 Colaboradores em acompanhamento")

df_view = df_f.sort_values("TEMPO DE CASA")

st.dataframe(df_view, use_container_width=True)

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Painel SKAP - Gestão de Desenvolvimento")

# =========================
# MODO ADMIN
# =========================
admin = st.sidebar.text_input("", type="password", placeholder="Modo administrador")

modo_admin = admin == "rhadmin"


modo_admin = admin == "rhadmin"


# =========================
# Utils
# =========================
def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    return df

def limpar_texto(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def garantir_coluna(df: pd.DataFrame, col: str, default="") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df

def consolidar_xy(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    if nome in df.columns:
        return df
    x = f"{nome}_X"
    y = f"{nome}_Y"
    if x in df.columns and y in df.columns:
        df[nome] = df[x]
        vazio = df[nome].astype(str).str.strip().isin(["", "nan", "None"])
        df.loc[vazio, nome] = df.loc[vazio, y]
        df.drop(columns=[x, y], inplace=True)
    elif x in df.columns:
        df[nome] = df[x]
        df.drop(columns=[x], inplace=True)
    elif y in df.columns:
        df[nome] = df[y]
        df.drop(columns=[y], inplace=True)
    else:
        df[nome] = ""
    return df

def tratar_data_adm(df: pd.DataFrame, col_data: str = "DATA ULT. ADM") -> pd.DataFrame:
    df = garantir_coluna(df, col_data, "")

    df[f"{col_data}_RAW"] = df[col_data]

    # 1) parse como texto BR
    dt_txt = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)

    # 2) parse como serial Excel, mas só para valores plausíveis
    num = pd.to_numeric(df[col_data], errors="coerce")

    # intervalo plausível de serial Excel (dias desde 1899-12-30):
    # ~1954-10-17 = 20000 | ~2119-01-01 = ~80000
    num_ok = num.where((num >= 20000) & (num <= 80000))

    dt_excel = pd.to_datetime(num_ok, unit="D", origin="1899-12-30", errors="coerce")

    df[col_data] = dt_txt.fillna(dt_excel)

    hoje = pd.to_datetime(datetime.today().date())
    df["TEMPO DE CASA"] = (hoje - df[col_data]).dt.days
    df["TEMPO DE CASA"] = df["TEMPO DE CASA"].fillna(0).astype(int)

    df["DATA ADMISSAO"] = df[col_data].dt.strftime("%d/%m/%Y").fillna("")
    return df

def opcoes(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    vals = (
        df[col].astype(str).str.strip()
        .replace(["", "nan", "None"], pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(vals)

# =========================
# Upload + recarregar
# =========================
if modo_admin:
    st.sidebar.header("📥 Atualização de dados")





if modo_admin:
    upload_skap = st.sidebar.file_uploader("Enviar Skap.xlsx", type=["xlsx"])
    upload_com = st.sidebar.file_uploader("Enviar Skap - comentarios.xlsx", type=["xlsx"])
else:
    upload_skap = None
    upload_com = None


if modo_admin:
    st.sidebar.button("Atualizar dados")


@st.cache_data(show_spinner=False)
def carregar_dados(upload1, upload2):
    if upload1 is None or upload2 is None:
        raise ValueError("Envie os 2 arquivos: Skap.xlsx e Skap - comentarios.xlsx.")
    skap_df = pd.read_excel(upload1)
    com_df = pd.read_excel(upload2)
    return skap_df, com_df

try:
    skap, comentarios = carregar_dados(upload_skap, upload_com)
except Exception as e:
    st.info("➡️ Envie os 2 arquivos no menu lateral para carregar o painel.")
    st.stop()

# =========================
# Normalização + colunas mínimas
# =========================
skap = normalizar_colunas(skap)
comentarios = normalizar_colunas(comentarios)

for col in ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA", "DATA ULT. ADM"]:
    skap = garantir_coluna(skap, col, "")

skap = limpar_texto(skap, ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA"])
comentarios = limpar_texto(comentarios, ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA"])

# =========================
# Datas + métricas
# =========================
skap = tratar_data_adm(skap, "DATA ULT. ADM")

# Percentuais
for col in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS"]:
    skap = garantir_coluna(skap, col, 0)
    skap[col] = pd.to_numeric(skap[col], errors="coerce").fillna(0)

# Prazos
skap["PRAZO TECNICAS"] = (skap["DATA ULT. ADM"] + pd.Timedelta(days=30)).dt.strftime("%d/%m/%Y").fillna("")
skap["PRAZO ESPECIFICAS"] = (skap["DATA ULT. ADM"] + pd.Timedelta(days=60)).dt.strftime("%d/%m/%Y").fillna("")

# Status
def status_tecnicas(row):
    if row["HABILIDADES TECNICAS"] > 0:
        return "Realizado"
    if row["TEMPO DE CASA"] <= 30:
        return "No prazo"
    return "Não realizado"

def status_especificas(row):
    if row["HABILIDADES ESPECIFICAS"] > 0:
        return "Realizado"
    if row["TEMPO DE CASA"] <= 60:
        return "No prazo"
    return "Não realizado"

skap["STATUS TECNICAS"] = skap.apply(status_tecnicas, axis=1)
skap["STATUS ESPECIFICAS"] = skap.apply(status_especificas, axis=1)

# =========================
# Merge com comentários
# =========================
base = skap.merge(comentarios, on="COLABORADOR", how="left", suffixes=("_X", "_Y")).fillna("")

for c in ["CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA"]:
    base = consolidar_xy(base, c)

# =========================
# Ordem das colunas
# =========================
ordem = [
    "COLABORADOR",
    "CARGO",
    "OPERACAO",
    "ATIVIDADE",
    "LIDERANCA",
    "DATA ADMISSAO",
    "TEMPO DE CASA",
    "HABILIDADES TECNICAS",
    "PRAZO TECNICAS",
    "STATUS TECNICAS",
    "HABILIDADES ESPECIFICAS",
    "PRAZO ESPECIFICAS",
    "STATUS ESPECIFICAS",
    "HABILIDADE TECNICA",
    "HABILIDADE ESPECIFICA",
    "HABILIDADE EMPODERAMENTO",
]
ordem = [c for c in ordem if c in base.columns]
base = base[ordem].copy()

# =========================
# Filtros
# =========================
st.sidebar.header("Filtros")

f_operacao = st.sidebar.multiselect("Operação", opcoes(base, "OPERACAO"))
f_lideranca = st.sidebar.multiselect("Liderança", opcoes(base, "LIDERANCA"))
f_atividade = st.sidebar.multiselect("Atividade", opcoes(base, "ATIVIDADE"))
f_status = st.sidebar.multiselect("Status", ["Realizado", "Não realizado", "No prazo"])

base_f = base.copy()
if f_operacao:
    base_f = base_f[base_f["OPERACAO"].isin(f_operacao)]
if f_lideranca:
    base_f = base_f[base_f["LIDERANCA"].isin(f_lideranca)]
if f_atividade:
    base_f = base_f[base_f["ATIVIDADE"].isin(f_atividade)]
if f_status:
    base_f = base_f[
        base_f["STATUS TECNICAS"].isin(f_status) |
        base_f["STATUS ESPECIFICAS"].isin(f_status)
    ]

# =========================
# Cards
# =========================
total = len(base_f)
pend = len(base_f[
    (base_f["STATUS TECNICAS"] == "Não realizado") |
    (base_f["STATUS ESPECIFICAS"] == "Não realizado")
])
concl = len(base_f[
    (base_f["STATUS TECNICAS"] == "Realizado") &
    (base_f["STATUS ESPECIFICAS"] == "Realizado")
])
nop = len(base_f[
    (base_f["STATUS TECNICAS"] == "No prazo") |
    (base_f["STATUS ESPECIFICAS"] == "No prazo")
])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", total)
c2.metric("✅ 100% concluídos", concl)
c3.metric("🟡 No prazo", nop)
c4.metric("🔴 Com pendência", pend)

st.divider()

# =========================
# Gráficos (decrescente)
# =========================
st.subheader("🔴 Pendências (Não realizado) por Operação")
gop = (
    base_f[
        (base_f["STATUS TECNICAS"] == "Não realizado") |
        (base_f["STATUS ESPECIFICAS"] == "Não realizado")
    ]
    .groupby("OPERACAO", dropna=False)
    .size()
    .sort_values(ascending=False)
    .reset_index(name="Quantidade")
)
fig1 = px.bar(gop, x="OPERACAO", y="Quantidade", text="Quantidade")
fig1.update_layout(xaxis={"categoryorder": "total descending"})
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🔴 Pendências (Não realizado) por Liderança")
glid = (
    base_f[
        (base_f["STATUS TECNICAS"] == "Não realizado") |
        (base_f["STATUS ESPECIFICAS"] == "Não realizado")
    ]
    .groupby("LIDERANCA", dropna=False)
    .size()
    .sort_values(ascending=False)
    .reset_index(name="Quantidade")
)
fig2 = px.bar(glid, x="LIDERANCA", y="Quantidade", text="Quantidade")
fig2.update_layout(xaxis={"categoryorder": "total descending"})
st.plotly_chart(fig2, use_container_width=True)

# =========================
# Tabela formatada
# =========================
def cor_status(val):
    if val == "Não realizado":
        return "color: red; font-weight: bold;"
    if val == "Realizado":
        return "color: green; font-weight: bold;"
    if val == "No prazo":
        return "color: orange; font-weight: bold;"
    return ""

styled = (
    base_f.style
    .format({
        "HABILIDADES TECNICAS": "{:.0%}",
        "HABILIDADES ESPECIFICAS": "{:.0%}",
    })
    .background_gradient(
        subset=[c for c in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS"] if c in base_f.columns],
        cmap="RdYlGn"
    )
    .applymap(cor_status, subset=["STATUS TECNICAS", "STATUS ESPECIFICAS"])
)

st.subheader("📋 Detalhamento Individual")
st.dataframe(styled, use_container_width=True)

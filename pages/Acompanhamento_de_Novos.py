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
# Garantir colunas (para não quebrar se faltar algo)
# =========================
def garantir_coluna(df_, col, default=""):
    if col not in df_.columns:
        df_[col] = default
    return df_

for col in [
    "COLABORADOR", "OPERACAO", "ATIVIDADE", "ADMISSAO", "TEMPO DE CASA", "PROGRESSO GERAL",
    "REACAO INTEGRACAO", "DT REACAO INTEGRACAO", "LIMITE REACAO INTEGRACAO"
]:
    df = garantir_coluna(df, col, "")

# =========================
# Datas e tempo de casa
# =========================
# Datas principais (admissão e reação)
df["ADMISSAO"] = pd.to_datetime(df["ADMISSAO"], errors="coerce", dayfirst=True)
df["DT REACAO INTEGRACAO"] = pd.to_datetime(df["DT REACAO INTEGRACAO"], errors="coerce", dayfirst=True)
df["LIMITE REACAO INTEGRACAO"] = pd.to_datetime(df["LIMITE REACAO INTEGRACAO"], errors="coerce", dayfirst=True)

hoje = pd.to_datetime(datetime.today().date())

# Tempo de casa
td = pd.to_numeric(df["TEMPO DE CASA"], errors="coerce")
if td.isna().all():
    df["TEMPO DE CASA"] = (hoje - df["ADMISSAO"]).dt.days
else:
    df["TEMPO DE CASA"] = td

df["TEMPO DE CASA"] = pd.to_numeric(df["TEMPO DE CASA"], errors="coerce").fillna(0).astype(int)

# =========================
# Sidebar filtros (sem Regional / Empresa)
# =========================
st.sidebar.header("Filtros")

def opcoes(col):
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

f_operacao = st.sidebar.multiselect("Operação", opcoes("OPERACAO"))
f_atividade = st.sidebar.multiselect("Atividade", opcoes("ATIVIDADE"))

df_f = df.copy()

if f_operacao:
    df_f = df_f[df_f["OPERACAO"].isin(f_operacao)]
if f_atividade:
    df_f = df_f[df_f["ATIVIDADE"].isin(f_atividade)]

# =========================
# Coluna DIAS (regra que você pediu)
# =========================
# Se Dt Reação Integração vazia -> (Limite - hoje)
# Se tiver valor -> (Dt - Limite)
dt = df_f["DT REACAO INTEGRACAO"]
lim = df_f["LIMITE REACAO INTEGRACAO"]

dias = np.where(
    dt.isna(),
    (lim - hoje).dt.days,
    (dt - lim).dt.days
)

df_f["DIAS"] = pd.to_numeric(dias, errors="coerce")

# =========================
# Cards (opcional: mantive simples)
# =========================
total = len(df_f)
c1, c2 = st.columns(2)
c1.metric("Total", total)
c2.metric("Média Tempo de Casa", int(df_f["TEMPO DE CASA"].mean()) if total else 0)

st.divider()

# =========================
# Ordem das colunas (exatamente como você pediu)
# =========================
df_view = df_f.copy()

# versões texto das datas para mostrar bonito
df_view["ADMISSAO"] = pd.to_datetime(df_view["ADMISSAO"], errors="coerce").dt.strftime("%d/%m/%Y").fillna("")
df_view["LIMITE REACAO INTEGRACAO"] = pd.to_datetime(df_view["LIMITE REACAO INTEGRACAO"], errors="coerce").dt.strftime("%d/%m/%Y").fillna("")
df_view["DT REACAO INTEGRACAO"] = pd.to_datetime(df_view["DT REACAO INTEGRACAO"], errors="coerce").dt.strftime("%d/%m/%Y").fillna("")

ordem = [
    "COLABORADOR",
    "OPERACAO",
    "ATIVIDADE",
    "ADMISSAO",
    "TEMPO DE CASA",
    "PROGRESSO GERAL",
    "REACAO INTEGRACAO",
    "LIMITE REACAO INTEGRACAO",
    "DT REACAO INTEGRACAO",
    "DIAS",
]
ordem = [c for c in ordem if c in df_view.columns]
df_view = df_view[ordem].copy()

# =========================
# Cor da coluna DIAS
# =========================
def cor_dias(v):
    try:
        v = float(v)
    except Exception:
        return ""
    if v > 0:
        return "color: #00c853; font-weight: 700;"  # verde
    if v < 0:
        return "color: #ff1744; font-weight: 700;"  # vermelho
    return "font-weight: 700;"

st.subheader("📋 Detalhamento — Reação Integração")

# Para garantir que o estilo apareça, usamos Styler
st.dataframe(
    df_view.style.applymap(cor_dias, subset=["DIAS"]),
    use_container_width=True
)

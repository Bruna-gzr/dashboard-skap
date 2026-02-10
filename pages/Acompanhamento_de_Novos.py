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
# Utils
# =========================
def garantir_coluna(df_, col, default=""):
    if col not in df_.columns:
        df_[col] = default
    return df_

def opcoes(df_, col):
    if col not in df_.columns:
        return []
    vals = (
        df_[col].astype(str).str.strip()
        .replace(["", "nan", "None"], pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(vals)

def centralizar_styler(df_):
    return df_.style.set_properties(**{"text-align": "center"}) \
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])

def to_datetime_safe(s: pd.Series) -> pd.Series:
    # texto BR + serial excel (blindado)
    dt_txt = pd.to_datetime(s, errors="coerce", dayfirst=True)

    num = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = num.notna() & np.isfinite(num) & (num >= 20000) & (num <= 80000)
    dt_excel = pd.Series(pd.NaT, index=s.index)
    if mask.any():
        dt_excel.loc[mask] = pd.to_datetime(
            num.loc[mask].astype("int64"),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return dt_txt.fillna(dt_excel)

# =========================
# Garantir colunas necessárias
# =========================
for col in [
    "COLABORADOR", "OPERACAO", "ATIVIDADE", "ADMISSAO", "TEMPO DE CASA",
    "PROGRESSO GERAL", "REACAO INTEGRACAO", "DT REACAO INTEGRACAO", "LIMITE REACAO INTEGRACAO"
]:
    df = garantir_coluna(df, col, "")

# =========================
# Datas + tempo de casa
# =========================
df["ADMISSAO"] = to_datetime_safe(df["ADMISSAO"])
df["DT REACAO INTEGRACAO"] = to_datetime_safe(df["DT REACAO INTEGRACAO"])
df["LIMITE REACAO INTEGRACAO"] = to_datetime_safe(df["LIMITE REACAO INTEGRACAO"])

hoje = pd.to_datetime(datetime.today().date())

td = pd.to_numeric(df["TEMPO DE CASA"], errors="coerce")
if td.isna().all():
    df["TEMPO DE CASA"] = (hoje - df["ADMISSAO"]).dt.days
else:
    df["TEMPO DE CASA"] = td
df["TEMPO DE CASA"] = pd.to_numeric(df["TEMPO DE CASA"], errors="coerce").fillna(0).astype(int)

# Progresso geral como número (0..1 ou 0..100)
pg = pd.to_numeric(df["PROGRESSO GERAL"], errors="coerce")
# se vier em 0..100, normaliza para 0..1
df["PROGRESSO_GERAL_NUM"] = np.where(pg.notna() & (pg > 1.0), pg / 100.0, pg).astype(float)
df["PROGRESSO_GERAL_NUM"] = np.nan_to_num(df["PROGRESSO_GERAL_NUM"], nan=0.0)

# =========================
# Sidebar filtros (somente Operação / Atividade)
# =========================
st.sidebar.header("Filtros")
f_operacao = st.sidebar.multiselect("Operação", opcoes(df, "OPERACAO"))
f_atividade = st.sidebar.multiselect("Atividade", opcoes(df, "ATIVIDADE"))

df_f = df.copy()
if f_operacao:
    df_f = df_f[df_f["OPERACAO"].isin(f_operacao)]
if f_atividade:
    df_f = df_f[df_f["ATIVIDADE"].isin(f_atividade)]

# =========================
# Coluna DIAS (regra NOVA)
# Se DT vazia -> HOJE - LIMITE
# Se DT preenchida -> LIMITE - DT
# =========================
dt = df_f["DT REACAO INTEGRACAO"]
lim = df_f["LIMITE REACAO INTEGRACAO"]

dias = np.where(
    dt.isna(),
    (hoje - lim).dt.days,
    (lim - dt).dt.days
)

df_f["DIAS"] = pd.to_numeric(dias, errors="coerce")

# =========================
# Cards
# =========================
total = len(df_f)
c1, c2 = st.columns(2)
c1.metric("Total", total)
c2.metric("Média Tempo de Casa", int(df_f["TEMPO DE CASA"].mean()) if total else 0)

st.divider()

# =========================
# Preparar visual (datas em texto + %)
# =========================
df_view = df_f.copy()

df_view["ADMISSAO"] = pd.to_datetime(df_view["ADMISSAO"], errors="coerce").dt.strftime("%d/%m/%Y").fillna("")
df_view["LIMITE REACAO INTEGRACAO"] = pd.to_datetime(df_view["LIMITE REACAO INTEGRACAO"], errors="coerce").dt.strftime("%d/%m/%Y").fillna("")
df_view["DT REACAO INTEGRACAO"] = pd.to_datetime(df_view["DT REACAO INTEGRACAO"], errors="coerce").dt.strftime("%d/%m/%Y").fillna("")

# Progresso em %
df_view["PROGRESSO GERAL"] = df_view["PROGRESSO_GERAL_NUM"].map(lambda x: f"{x:.0%}")

# Ordem das colunas
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
df_view = df_view[[c for c in ordem if c in df_view.columns]].copy()

# =========================
# Estilos (cores)
# =========================
def estilo_dias(v):
    try:
        v = float(v)
    except Exception:
        return "text-align: center;"
    # (mantive o padrão que você tinha antes: >0 verde, <0 vermelho)
    # com a NOVA regra, pode acontecer de "faltando" ser negativo e "atraso" positivo.
    if v > 0:
        return "color: #00c853; font-weight: 700; text-align: center;"
    if v < 0:
        return "color: #ff1744; font-weight: 700; text-align: center;"
    return "font-weight: 700; text-align: center;"

def estilo_progresso(v):
    # v vem como "90%" etc
    try:
        n = float(str(v).replace("%", "").strip())
    except Exception:
        return "text-align: center;"
    if n >= 100:
        return "color: #00c853; font-weight: 700; text-align: center;"  # verde
    if n >= 80:
        return "color: #ffd600; font-weight: 700; text-align: center;"  # amarelo
    return "color: #ff1744; font-weight: 700; text-align: center;"      # vermelho

def estilo_status(v):
    s = str(v).strip().lower()

    if s == "realizada":
        return "color: #00c853; font-weight: 700; text-align: center;"   # verde
    if s == "não realizada" or s == "nao realizada":
        return "color: #ff1744; font-weight: 700; text-align: center;"   # vermelho
    if s == "realizada - fora do prazo":
        return "color: #ff9100; font-weight: 700; text-align: center;"   # laranja
    if s == "no prazo":
        return "color: #ffd600; font-weight: 700; text-align: center;"   # amarelo
    if s == "n/a" or s == "na":
        return "color: #ffffff; font-weight: 700; text-align: center;"   # branco

    return "text-align: center;"

# =========================
# Render
# =========================
st.subheader("📋 Detalhamento — Reação Integração")

styler = df_view.style

# centralizar tudo
styler = styler.set_properties(**{"text-align": "center"})
styler = styler.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])

# cores por coluna
if "DIAS" in df_view.columns:
    styler = styler.applymap(estilo_dias, subset=["DIAS"])
if "PROGRESSO GERAL" in df_view.columns:
    styler = styler.applymap(estilo_progresso, subset=["PROGRESSO GERAL"])
if "REACAO INTEGRACAO" in df_view.columns:
    styler = styler.applymap(estilo_status, subset=["REACAO INTEGRACAO"])

st.dataframe(
    styler,
    use_container_width=True,
    height=700
)

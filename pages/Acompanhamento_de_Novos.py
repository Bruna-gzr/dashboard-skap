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

def fmt_data(d: pd.Series) -> pd.Series:
    # dd/mm/yyyy (vazio quando NaT)
    return pd.to_datetime(d, errors="coerce").dt.strftime("%d/%m/%Y").fillna("")

def fmt_datahora(d: pd.Series) -> pd.Series:
    # dd/mm/yyyy HH:MM (se tiver horário; se não tiver, mostra dd/mm/yyyy 00:00)
    x = pd.to_datetime(d, errors="coerce")
    return x.dt.strftime("%d/%m/%Y %H:%M").fillna("")

# =========================
# Garantir colunas básicas
# =========================
base_cols = [
    "COLABORADOR", "OPERACAO", "ATIVIDADE", "ADMISSAO", "TEMPO DE CASA", "PROGRESSO GERAL"
]
for c in base_cols:
    df = garantir_coluna(df, c, "")

# Garantir colunas das etapas
etapas = [
    # (nome_da_aba, status_col, limite_col, dt_col, dt_e_datahora?)
    ("Reação Integração", "REACAO INTEGRACAO", "LIMITE REACAO INTEGRACAO", "DT REACAO INTEGRACAO", False),
    ("Satisfação I", "SATISFACAO COM A INTEGRACAO I", "LIMITE SATISF. I", "DT HR SATISF. I", True),
    ("AVD", "AVD", "LIMITE AVD", "DT HR AVD", True),
    ("Feedback AVD", "FEEDBACK AVD", "LIMITE FEEDBACK", "DT HR FEEDBACK", True),
    ("Cadastro PDI", "CADASTRO PDI", "LIMITE PDI", "DT PDI", False),
    ("Satisfação II", "SATISFACAO COM A INTEGRACAO II", "LIMITE SATISFACAO II", "DT SATISFACAO II", False),
    ("Follow", "FOLLOW", "LIMITE FOLLOW", "DATA HR FOLLOW", True),
]

for _, status_col, limite_col, dt_col, _ in etapas:
    df = garantir_coluna(df, status_col, "")
    df = garantir_coluna(df, limite_col, "")
    df = garantir_coluna(df, dt_col, "")

# =========================
# Datas + tempo de casa
# =========================
df["ADMISSAO"] = to_datetime_safe(df["ADMISSAO"])
hoje = pd.to_datetime(datetime.today().date())

td = pd.to_numeric(df["TEMPO DE CASA"], errors="coerce")
if td.isna().all():
    df["TEMPO DE CASA"] = (hoje - df["ADMISSAO"]).dt.days
else:
    df["TEMPO DE CASA"] = td
df["TEMPO DE CASA"] = pd.to_numeric(df["TEMPO DE CASA"], errors="coerce").fillna(0).astype(int)

# Progresso geral como número (0..1 ou 0..100)
pg = pd.to_numeric(df["PROGRESSO GERAL"], errors="coerce")
df["PROGRESSO_GERAL_NUM"] = np.where(pg.notna() & (pg > 1.0), pg / 100.0, pg).astype(float)
df["PROGRESSO_GERAL_NUM"] = np.nan_to_num(df["PROGRESSO_GERAL_NUM"], nan=0.0)

# Converter colunas de limite/dt para datetime
for _, _, limite_col, dt_col, _ in etapas:
    df[limite_col] = to_datetime_safe(df[limite_col])
    df[dt_col] = to_datetime_safe(df[dt_col])

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
# Cards
# =========================
total = len(df_f)
c1, c2 = st.columns(2)
c1.metric("Total", total)
c2.metric("Média Tempo de Casa", int(df_f["TEMPO DE CASA"].mean()) if total else 0)

st.divider()

# =========================
# Estilos (cores + centralização)
# =========================
def estilo_progresso(v):
    # v vem como "90%"
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
    if s in ["não realizada", "nao realizada"]:
        return "color: #ff1744; font-weight: 700; text-align: center;"   # vermelho
    if s == "realizada - fora do prazo":
        return "color: #ff9100; font-weight: 700; text-align: center;"   # laranja
    if s == "no prazo":
        return "color: #ffd600; font-weight: 700; text-align: center;"   # amarelo
    if s in ["n/a", "na"]:
        return "color: #ffffff; font-weight: 700; text-align: center;"   # branco

    return "text-align: center;"

def estilo_dias(v):
    try:
        v = int(v)
    except Exception:
        return "text-align: center;"
    # Como DIAS = LIMITE - HOJE (ou LIMITE - DT):
    # positivo = ainda faltam dias -> verde
    # negativo = passou do limite -> vermelho
    if v > 0:
        return "color: #00c853; font-weight: 700; text-align: center;"
    if v < 0:
        return "color: #ff1744; font-weight: 700; text-align: center;"
    return "color: #ffd600; font-weight: 700; text-align: center;"  # 0 = vence hoje (amarelo)

# =========================
# Função para montar tabela por etapa
# =========================
def tabela_etapa(nome_aba, status_col, limite_col, dt_col, dt_e_datahora: bool):
    tmp = df_f.copy()

    # DIAS: se dt vazia -> limite - hoje; se dt preenchida -> limite - dt
    dt = tmp[dt_col]
    lim = tmp[limite_col]

    dias = np.where(
    dt.isna(),
    (lim - hoje).dt.days,
    (lim - dt).dt.days
)

# robusto: converte, mantém NaN, e usa inteiro sem quebrar
tmp["DIAS"] = pd.to_numeric(pd.Series(dias), errors="coerce")
tmp["DIAS"] = tmp["DIAS"].round(0)
tmp["DIAS"] = tmp["DIAS"].astype("Int64")  # permite <NA>


    # Formata datas para exibição
    tmp["ADMISSAO"] = fmt_data(tmp["ADMISSAO"])
    tmp[limite_col] = fmt_data(tmp[limite_col])
    tmp[dt_col] = fmt_datahora(tmp[dt_col]) if dt_e_datahora else fmt_data(tmp[dt_col])

    # Progresso em %
    tmp["PROGRESSO GERAL"] = tmp["PROGRESSO_GERAL_NUM"].map(lambda x: f"{x:.0%}")

    # Ordem das colunas (base + etapa + Dias)
    view = tmp[
        [
            "COLABORADOR",
            "OPERACAO",
            "ATIVIDADE",
            "ADMISSAO",
            "TEMPO DE CASA",
            "PROGRESSO GERAL",
            status_col,
            limite_col,
            dt_col,
            "DIAS",
        ]
    ].copy()

    # Styler: centralização + cores
    styler = view.style
    styler = styler.set_properties(**{"text-align": "center"})
    styler = styler.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])

    styler = styler.applymap(estilo_progresso, subset=["PROGRESSO GERAL"])
    styler = styler.applymap(estilo_status, subset=[status_col])
    styler = styler.applymap(estilo_dias, subset=["DIAS"])

    st.subheader(f"📋 Detalhamento — {nome_aba}")
    st.dataframe(styler, use_container_width=True, height=700)

# =========================
# Abas
# =========================
abas = st.tabs([e[0] for e in etapas])

for tab, (nome_aba, status_col, limite_col, dt_col, dt_e_datahora) in zip(abas, etapas):
    with tab:
        tabela_etapa(nome_aba, status_col, limite_col, dt_col, dt_e_datahora)

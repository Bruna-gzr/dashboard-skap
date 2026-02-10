import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from io import BytesIO
import plotly.express as px

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

# ✅ mais robusta: limpa texto, aceita NBSP, hífen, serial excel com fração de dia
def to_datetime_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    s2 = s.copy()

    s_str = (
        s2.astype(str)
        .str.replace("\u00a0", " ", regex=False)  # NBSP (espaço invisível)
        .str.strip()
    )
    s_str = s_str.replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaT": pd.NA, "-": pd.NA}
    )

    # 1) tenta parse texto BR
    dt_txt = pd.to_datetime(s_str, errors="coerce", dayfirst=True)

    # 2) tenta serial excel (inclui fração de dia)
    num = pd.to_numeric(s_str, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = num.notna() & np.isfinite(num) & (num >= 20000) & (num <= 80000)

    dt_excel = pd.Series(pd.NaT, index=s2.index)
    if mask.any():
        dt_excel.loc[mask] = pd.to_datetime(
            num.loc[mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return dt_txt.fillna(dt_excel)

def fmt_data(d: pd.Series) -> pd.Series:
    # sempre dd/mm/aaaa (sem hora)
    return pd.to_datetime(d, errors="coerce").dt.strftime("%d/%m/%Y").fillna("")

def normalizar_status(s) -> str:
    s = str(s).strip()
    if s.lower() in ["nan", "none", ""]:
        return ""

    sl = s.lower()

    if sl in ["realizada", "realizado"]:
        return "Realizada"

    if sl in ["não realizada", "nao realizada", "não realizado", "nao realizado"]:
        return "Não Realizada"

    if sl in ["realizada - fora do prazo", "realizado - fora do prazo"]:
        return "Realizada - Fora do Prazo"

    if sl == "no prazo":
        return "No prazo"

    if sl in ["n/a", "na"]:
        return "N/A"

    return s

def preparar_excel_para_download(df_: pd.DataFrame, sheet_name="Dados") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()

# =========================
# Colunas e etapas
# =========================
base_cols = [
    "COLABORADOR", "OPERACAO", "ATIVIDADE", "ADMISSAO", "TEMPO DE CASA", "PROGRESSO GERAL"
]
for c in base_cols:
    df = garantir_coluna(df, c, "")

etapas = [
    ("Reação Integração", "REACAO INTEGRACAO", "LIMITE REACAO INTEGRACAO", "DT REACAO INTEGRACAO"),
    ("Satisfação I", "SATISFACAO COM A INTEGRACAO I", "LIMITE SATISF. I", "DT HR SATISF. I"),
    ("AVD", "AVD", "LIMITE AVD", "DT HR AVD"),
    ("Feedback AVD", "FEEDBACK AVD", "LIMITE FEEDBACK", "DT HR FEEDBACK"),
    ("Cadastro PDI", "CADASTRO PDI", "LIMITE PDI", "DT PDI"),
    ("Satisfação II", "SATISFACAO COM A INTEGRACAO II", "LIMITE SATISFACAO II", "DT SATISFACAO II"),
    ("Follow", "FOLLOW", "LIMITE FOLLOW", "DATA HR FOLLOW"),
]

for _, status_col, limite_col, dt_col in etapas:
    df = garantir_coluna(df, status_col, "")
    df = garantir_coluna(df, limite_col, "")
    df = garantir_coluna(df, dt_col, "")

# =========================
# Datas + tempo de casa
# =========================
df["ADMISSAO"] = to_datetime_safe(df["ADMISSAO"])
hoje = pd.to_datetime(datetime.today().date())

# Ignorar admitidos antes de 01/01/2025
cutoff = pd.to_datetime("2025-01-01")
df = df[df["ADMISSAO"].notna() & (df["ADMISSAO"] >= cutoff)].copy()

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

# Converter limite/dt para datetime
for _, _, limite_col, dt_col in etapas:
    df[limite_col] = to_datetime_safe(df[limite_col])
    df[dt_col] = to_datetime_safe(df[dt_col])

# Normalizar status
for _, status_col, _, _ in etapas:
    df[status_col] = df[status_col].apply(normalizar_status)

# =========================
# Sidebar filtros (Operação / Atividade / Status / Período Admissão)
# =========================
st.sidebar.header("Filtros")

f_operacao = st.sidebar.multiselect("Operação", opcoes(df, "OPERACAO"))
f_atividade = st.sidebar.multiselect("Atividade", opcoes(df, "ATIVIDADE"))
f_status = st.sidebar.multiselect(
    "Status (aplica na etapa da aba)",
    ["Realizada", "Não Realizada", "Realizada - Fora do Prazo", "N/A", "No prazo"]
)

# filtro por período de admissão
min_adm = df["ADMISSAO"].min()
max_adm = df["ADMISSAO"].max()
if pd.isna(min_adm) or pd.isna(max_adm):
    min_adm = pd.to_datetime("2025-01-01")
    max_adm = hoje

dt_ini, dt_fim = st.sidebar.date_input(
    "Período de admissão",
    value=(min_adm.date(), max_adm.date()),
)

dt_ini = pd.to_datetime(dt_ini)
dt_fim = pd.to_datetime(dt_fim)
if dt_fim < dt_ini:
    dt_ini, dt_fim = dt_fim, dt_ini

df_f = df.copy()
df_f = df_f[df_f["ADMISSAO"].between(dt_ini, dt_fim)].copy()

if f_operacao:
    df_f = df_f[df_f["OPERACAO"].isin(f_operacao)]
if f_atividade:
    df_f = df_f[df_f["ATIVIDADE"].isin(f_atividade)]

# ✅ DEBUG opcional (não aparece a menos que você marque)
if st.sidebar.checkbox("DEBUG: checar datas que viraram NaT", value=False):
    st.write("Linhas filtradas:", len(df_f))
    for nome_aba, status_col, limite_col, dt_col in etapas:
        st.write(f"### {nome_aba}")
        st.write("Limite NaT:", int(df_f[limite_col].isna().sum()), "de", len(df_f))
        st.write("Dt NaT:", int(df_f[dt_col].isna().sum()), "de", len(df_f))
        exemplo = df_f[df_f[limite_col].isna()][["COLABORADOR", status_col, limite_col, dt_col]].head(5)
        st.dataframe(exemplo, use_container_width=True)

# =========================
# Cards globais
# =========================
no_prazo_vencendo_ids = set()
nao_realizada_ids = set()

for _, status_col, limite_col, _ in etapas:
    st_col = df_f[status_col].fillna("")
    lim = df_f[limite_col]

    # No prazo vencendo em até 3 dias
    mask_np = (st_col == "No prazo") & lim.notna()
    dias_para_vencer = (lim - hoje).dt.days
    mask_venc3 = mask_np & (dias_para_vencer >= 0) & (dias_para_vencer <= 3)
    no_prazo_vencendo_ids.update(df_f.loc[mask_venc3, "COLABORADOR"].astype(str).tolist())

    # Não Realizada em alguma etapa
    mask_nr = (st_col == "Não Realizada")
    nao_realizada_ids.update(df_f.loc[mask_nr, "COLABORADOR"].astype(str).tolist())

c1, c2, c3 = st.columns(3)
c1.metric("Total (Admissão ≥ 01/01/2025)", len(df_f))
c2.metric("🟡 No prazo vencendo em até 3 dias", len(no_prazo_vencendo_ids))
c3.metric("🔴 Com alguma etapa Não Realizada", len(nao_realizada_ids))

st.divider()

# =========================
# Gráfico: % médio de Progresso Geral por Operação (em %)
# =========================
st.subheader("📈 Progresso Geral médio por Operação")

prog_op = (
    df_f.groupby("OPERACAO", dropna=False)["PROGRESSO_GERAL_NUM"]
    .mean()
    .sort_values(ascending=False)
    .reset_index(name="PROGRESSO_MEDIO")
)

if len(prog_op) == 0:
    st.info("Sem dados para exibir o gráfico com os filtros atuais.")
else:
    prog_op["PROGRESSO_MEDIO_%"] = (prog_op["PROGRESSO_MEDIO"] * 100).round(1)
    fig = px.bar(prog_op, x="OPERACAO", y="PROGRESSO_MEDIO_%", text="PROGRESSO_MEDIO_%")
    fig.update_layout(yaxis_title="% médio", xaxis_title="Operação", xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# Top 5 Operações com mais Não Realizada (global)
# =========================
st.subheader("🏆 Top 5 operações com mais 'Não Realizada' (geral)")

nr_counts = []
for _, status_col, _, _ in etapas:
    tmp = df_f[df_f[status_col] == "Não Realizada"].groupby("OPERACAO").size()
    nr_counts.append(tmp)

if nr_counts:
    nr_total = pd.concat(nr_counts, axis=0).groupby(level=0).sum().sort_values(ascending=False)
    top5_global = nr_total.head(5).reset_index()
    top5_global.columns = ["OPERACAO", "QTD_NAO_REALIZADA"]
else:
    top5_global = pd.DataFrame(columns=["OPERACAO", "QTD_NAO_REALIZADA"])

if len(top5_global) == 0:
    st.info("Nenhuma ocorrência de 'Não Realizada' com os filtros atuais.")
else:
    st.dataframe(top5_global, use_container_width=True)

st.divider()

# =========================
# Estilos
# =========================
def estilo_progresso(v):
    try:
        n = float(str(v).replace("%", "").strip())
    except Exception:
        return "text-align: center;"
    if n >= 100:
        return "color: #00c853; font-weight: 700; text-align: center;"
    if n >= 80:
        return "color: #ffd600; font-weight: 700; text-align: center;"
    return "color: #ff1744; font-weight: 700; text-align: center;"

def estilo_status(v):
    s = str(v).strip().lower()
    if s == "realizada":
        return "color: #00c853; font-weight: 700; text-align: center;"
    if s == "não realizada":
        return "color: #ff1744; font-weight: 700; text-align: center;"
    if s == "realizada - fora do prazo":
        return "color: #ff9100; font-weight: 700; text-align: center;"
    if s == "no prazo":
        return "color: #ffd600; font-weight: 700; text-align: center;"
    if s == "n/a":
        return "text-align: center;"
    return "text-align: center;"

def estilo_dias(v):
    try:
        v = int(v)
    except Exception:
        return "text-align: center;"
    # positivo = faltam dias; negativo = atrasado
    if v > 0:
        return "color: #00c853; font-weight: 700; text-align: center;"
    if v < 0:
        return "color: #ff1744; font-weight: 700; text-align: center;"
    return "color: #ffd600; font-weight: 700; text-align: center;"

# =========================
# Função por etapa
# =========================
def tabela_etapa(nome_aba, status_col, limite_col, dt_col):
    tmp = df_f.copy()

    # Aplicar filtro de status (apenas na etapa da aba)
    if f_status:
        tmp = tmp[tmp[status_col].isin(f_status)].copy()

    dt = tmp[dt_col]
    lim = tmp[limite_col]
    status = tmp[status_col].fillna("")

    # DIAS:
    # - padrão: limite - hoje
    # - se tiver dt preenchida: limite - dt
    # - se status == "No prazo": sempre limite - hoje
    dias = (lim - hoje).dt.days
    dias_dt = (lim - dt).dt.days
    dias = dias.where(dt.isna(), dias_dt)
    dias = dias.where(status != "No prazo", (lim - hoje).dt.days)

    # garante inteiro e nunca None
    tmp["DIAS"] = pd.to_numeric(dias, errors="coerce").fillna(0).astype(int)

    # Formatação: datas só com dd/mm/aaaa
    tmp["ADMISSAO"] = fmt_data(tmp["ADMISSAO"])
    tmp[limite_col] = fmt_data(tmp[limite_col])
    tmp[dt_col] = fmt_data(tmp[dt_col])

    tmp["PROGRESSO GERAL"] = tmp["PROGRESSO_GERAL_NUM"].map(lambda x: f"{x:.0%}")

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

    # Top 5 operações com mais "Não Realizada" NESTA etapa
    st.write("**Top 5 operações com mais 'Não Realizada' (nesta etapa)**")
    top5_etapa = (
        tmp[tmp[status_col] == "Não Realizada"]
        .groupby("OPERACAO")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .reset_index(name="QTD_NAO_REALIZADA")
    )

    if len(top5_etapa) == 0:
        st.caption("Sem 'Não Realizada' nesta etapa com os filtros atuais.")
    else:
        st.dataframe(top5_etapa, use_container_width=True)

    # Exportação: aba atual
    st.write("**Exportação (respeita filtros + etapa + status selecionado)**")
    excel_bytes = preparar_excel_para_download(view, sheet_name=nome_aba)
    st.download_button(
        label="⬇️ Baixar Excel desta etapa",
        data=excel_bytes,
        file_name=f"acomp_novos_{nome_aba.lower().replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.subheader(f"📋 Detalhamento — {nome_aba}")

    styler = view.style
    styler = styler.set_properties(**{
        "text-align": "center",
        "max-width": "140px",
        "white-space": "nowrap",
        "overflow": "hidden",
        "text-overflow": "ellipsis",
        "font-size": "12px",
    })
    styler = styler.set_table_styles([
        {"selector": "th", "props": [("text-align", "center"), ("font-size", "12px")]},
    ])

    styler = styler.applymap(estilo_progresso, subset=["PROGRESSO GERAL"])
    styler = styler.applymap(estilo_status, subset=[status_col])
    styler = styler.applymap(estilo_dias, subset=["DIAS"])

    st.dataframe(styler, use_container_width=True, height=700)

# =========================
# Abas
# =========================
abas = st.tabs([e[0] for e in etapas])

for tab, (nome_aba, status_col, limite_col, dt_col) in zip(abas, etapas):
    with tab:
        tabela_etapa(nome_aba, status_col, limite_col, dt_col)

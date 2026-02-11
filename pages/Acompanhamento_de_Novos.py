import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from io import BytesIO
import plotly.express as px
from zoneinfo import ZoneInfo

st.title("🆕 Acompanhamento de Novos")

# =========================
# Arquivo
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARQ_NOVOS = DATA_DIR / "Acomp novos.xlsx"

# =========================
# Última atualização dos dados (APENAS desta página)
# =========================
try:
    if ARQ_NOVOS.exists():
        last_mtime = ARQ_NOVOS.stat().st_mtime
        dt = datetime.fromtimestamp(last_mtime, tz=ZoneInfo("America/Sao_Paulo"))
        st.caption(f"🕒 Última atualização (dados): {dt.strftime('%d/%m/%Y %H:%M')}")
    else:
        st.caption("🕒 Última atualização (dados): arquivo não encontrado em /data")
except:
    st.caption("🕒 Última atualização (dados): não disponível")

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
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    s2 = s.copy()
    s_str = (
        s2.astype(str)
        .str.replace("\u00a0", " ", regex=False)  # NBSP
        .str.strip()
    )
    s_str = s_str.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaT": pd.NA, "-": pd.NA})

    dt_txt = pd.to_datetime(s_str, errors="coerce", dayfirst=True)

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
base_cols = ["COLABORADOR", "OPERACAO", "ATIVIDADE", "ADMISSAO", "TEMPO DE CASA", "PROGRESSO GERAL"]
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

# hoje no fuso BR (evita +1 dia no Streamlit Cloud)
hoje = pd.Timestamp.now(tz=ZoneInfo("America/Sao_Paulo")).normalize().tz_localize(None)

cutoff = pd.to_datetime("2025-01-01")
df = df[df["ADMISSAO"].notna() & (df["ADMISSAO"] >= cutoff)].copy()

td = pd.to_numeric(df["TEMPO DE CASA"], errors="coerce")
if td.isna().all():
    df["TEMPO DE CASA"] = (hoje - df["ADMISSAO"]).dt.days
else:
    df["TEMPO DE CASA"] = td
df["TEMPO DE CASA"] = pd.to_numeric(df["TEMPO DE CASA"], errors="coerce").fillna(0).astype(int)

pg = pd.to_numeric(df["PROGRESSO GERAL"], errors="coerce")
df["PROGRESSO_GERAL_NUM"] = np.where(pg.notna() & (pg > 1.0), pg / 100.0, pg).astype(float)
df["PROGRESSO_GERAL_NUM"] = np.nan_to_num(df["PROGRESSO_GERAL_NUM"], nan=0.0)

for _, _, limite_col, dt_col in etapas:
    df[limite_col] = to_datetime_safe(df[limite_col])
    df[dt_col] = to_datetime_safe(df[dt_col])

for _, status_col, _, _ in etapas:
    df[status_col] = df[status_col].apply(normalizar_status)

# =========================
# Sidebar filtros
# =========================
st.sidebar.header("Filtros")

f_operacao = st.sidebar.multiselect("Operação", opcoes(df, "OPERACAO"))
f_atividade = st.sidebar.multiselect("Atividade", opcoes(df, "ATIVIDADE"))
f_status = st.sidebar.multiselect(
    "Status (aplica na etapa da aba)",
    ["Realizada", "Não Realizada", "Realizada - Fora do Prazo", "N/A", "No prazo"]
)

min_adm = df["ADMISSAO"].min()
max_adm = df["ADMISSAO"].max()
if pd.isna(min_adm) or pd.isna(max_adm):
    min_adm = pd.to_datetime("2025-01-01")
    max_adm = hoje

dt_ini, dt_fim = st.sidebar.date_input("Período de admissão", value=(min_adm.date(), max_adm.date()))
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

# =========================
# Cards globais
# =========================
no_prazo_venc3_ids = set()
nao_realizada_ids = set()

for _, status_col, limite_col, _ in etapas:
    st_col = df_f[status_col].fillna("")
    lim = df_f[limite_col]

    # ✅ SOMENTE "No prazo"
    mask_np = (st_col == "No prazo") & lim.notna()
    dias_para_vencer = (lim - hoje).dt.days
    mask_venc3 = mask_np & dias_para_vencer.between(0, 3)
    no_prazo_venc3_ids.update(df_f.loc[mask_venc3, "COLABORADOR"].astype(str).tolist())

    mask_nr = (st_col == "Não Realizada")
    nao_realizada_ids.update(df_f.loc[mask_nr, "COLABORADOR"].astype(str).tolist())

c1, c2, c3 = st.columns(3)
c1.metric("Total (Admissão ≥ 01/01/2025)", len(df_f))
c2.metric("🟡 No prazo vencendo em até 3 dias", len(no_prazo_venc3_ids))
c3.metric("🔴 Com alguma etapa Não Realizada", len(nao_realizada_ids))

st.divider()

# =========================
# Progresso Geral (empresa) - barra
# =========================
st.subheader("📌 Aderência Média - Log20")

progresso_empresa = float(df_f["PROGRESSO_GERAL_NUM"].mean()) if len(df_f) else 0.0
progresso_empresa = max(0.0, min(1.0, progresso_empresa))  # garante 0..1

c_bar, c_txt = st.columns([6, 1])
with c_txt:
    st.markdown(f"### {progresso_empresa:.2%}")

with c_bar:
    st.progress(progresso_empresa)

# =========================
# Estilos (usados em tudo)
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
    if v > 0:
        return "color: #00c853; font-weight: 700; text-align: center;"
    if v < 0:
        return "color: #ff1744; font-weight: 700; text-align: center;"
    return "color: #ffd600; font-weight: 700; text-align: center;"

def styler_padrao(df_view: pd.DataFrame):
    sty = df_view.style
    sty = sty.set_properties(**{
        "text-align": "center",
        "max-width": "140px",
        "white-space": "nowrap",
        "overflow": "hidden",
        "text-overflow": "ellipsis",
        "font-size": "12px",
    })
    sty = sty.set_table_styles([
        {"selector": "th", "props": [("text-align", "center"), ("font-size", "12px")]},
    ])
    return sty

# =========================
# Gráfico: % médio de Progresso Geral por Operação
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
# 🔴 Não Realizada — detalhamento geral
# =========================
st.subheader("🔴 Não Realizada — detalhamento geral")

linhas_nr = []
for nome_aba, status_col, limite_col, dt_col in etapas:
    st_col = df_f[status_col].fillna("")
    lim = df_f[limite_col]
    dt = df_f[dt_col]

    mask_nr = (st_col == "Não Realizada") & lim.notna()
    if mask_nr.any():
        dias = (lim - hoje).dt.days
        dias_dt = (lim - dt).dt.days
        dias = dias.where(dt.isna(), dias_dt)

        tmp_nr = df_f.loc[mask_nr, ["COLABORADOR", "OPERACAO", "ATIVIDADE", "ADMISSAO", "TEMPO DE CASA"]].copy()
        tmp_nr["ETAPA"] = nome_aba
        tmp_nr["DATA LIMITE"] = lim.loc[mask_nr].copy()
        tmp_nr["DIAS"] = pd.to_numeric(dias.loc[mask_nr], errors="coerce").fillna(0).astype(int).values
        linhas_nr.append(tmp_nr)

if linhas_nr:
    df_nr = pd.concat(linhas_nr, ignore_index=True)
    df_nr = df_nr.sort_values(["DIAS", "ADMISSAO"], ascending=[True, True])  # mais crítico (negativo) primeiro
    df_nr["ADMISSAO"] = fmt_data(df_nr["ADMISSAO"])
    df_nr["DATA LIMITE"] = fmt_data(df_nr["DATA LIMITE"])

    st.metric("Registros com etapa 'Não Realizada'", len(df_nr))

    view_nr = df_nr[["COLABORADOR","OPERACAO","ATIVIDADE","ADMISSAO","TEMPO DE CASA","ETAPA","DATA LIMITE","DIAS"]].copy()
    sty_nr = styler_padrao(view_nr).applymap(estilo_dias, subset=["DIAS"])
    st.dataframe(sty_nr, use_container_width=True, height=450)
else:
    st.metric("Registros com etapa 'Não Realizada'", 0)
    st.info("Nenhuma ocorrência de 'Não Realizada' com os filtros atuais.")

st.divider()

# =========================
# 🟡 No prazo vencendo em até 3 dias (geral) — detalhamento
# =========================
st.subheader("🟡 No prazo vencendo em até 3 dias (geral)")

linhas_np = []
for nome_aba, status_col, limite_col, dt_col in etapas:
    st_col = df_f[status_col].fillna("")
    lim = df_f[limite_col]

    mask_np = (st_col == "No prazo") & lim.notna()
    dias_para_vencer = (lim - hoje).dt.days
    mask_venc3 = mask_np & dias_para_vencer.between(0, 3)

    if mask_venc3.any():
        tmp_np = df_f.loc[mask_venc3, ["COLABORADOR", "OPERACAO", "ATIVIDADE", "ADMISSAO", "TEMPO DE CASA"]].copy()
        tmp_np["ETAPA"] = nome_aba
        tmp_np["DATA LIMITE"] = lim.loc[mask_venc3].copy()
        tmp_np["DIAS"] = pd.to_numeric(dias_para_vencer.loc[mask_venc3], errors="coerce").fillna(0).astype(int).values
        linhas_np.append(tmp_np)

if linhas_np:
    df_alerta = pd.concat(linhas_np, ignore_index=True)
    df_alerta = df_alerta.sort_values(["DIAS", "ADMISSAO"], ascending=[True, True])
    df_alerta["ADMISSAO"] = fmt_data(df_alerta["ADMISSAO"])
    df_alerta["DATA LIMITE"] = fmt_data(df_alerta["DATA LIMITE"])

    st.metric("Registros 'No prazo' vencendo em até 3 dias", len(df_alerta))

    view_np = df_alerta[["COLABORADOR","OPERACAO","ATIVIDADE","ADMISSAO","TEMPO DE CASA","ETAPA","DATA LIMITE","DIAS"]].copy()
    sty_np = styler_padrao(view_np).applymap(estilo_dias, subset=["DIAS"])
    st.dataframe(sty_np, use_container_width=True, height=420)
else:
    st.metric("Registros 'No prazo' vencendo em até 3 dias", 0)
    st.info("Nenhum colaborador com etapa 'No prazo' vencendo em até 3 dias com os filtros atuais.")

st.divider()

# =========================
# Função por etapa (abas)
# =========================
def tabela_etapa(nome_aba, status_col, limite_col, dt_col):
    tmp = df_f.copy()
    tmp = tmp.sort_values("ADMISSAO", ascending=True)

    if f_status:
        tmp = tmp[tmp[status_col].isin(f_status)].copy()

    dt = tmp[dt_col]
    lim = tmp[limite_col]
    status = tmp[status_col].fillna("")

    dias = (lim - hoje).dt.days
    dias_dt = (lim - dt).dt.days
    dias = dias.where(dt.isna(), dias_dt)
    dias = dias.where(status != "No prazo", (lim - hoje).dt.days)

    tmp["DIAS"] = pd.to_numeric(dias, errors="coerce").fillna(0).astype(int)

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

    sty = styler_padrao(view)
    sty = sty.applymap(estilo_progresso, subset=["PROGRESSO GERAL"])
    sty = sty.applymap(estilo_status, subset=[status_col])
    sty = sty.applymap(estilo_dias, subset=["DIAS"])

    st.dataframe(sty, use_container_width=True, height=700)

# =========================
# Abas
# =========================
abas = st.tabs([e[0] for e in etapas])

for tab, (nome_aba, status_col, limite_col, dt_col) in zip(abas, etapas):
    with tab:
        tabela_etapa(nome_aba, status_col, limite_col, dt_col)

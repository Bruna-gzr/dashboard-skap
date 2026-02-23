import streamlit as st

st.set_page_config(
    page_title="Dashboard RH",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from io import BytesIO
import unicodedata
import plotly.express as px
from zoneinfo import ZoneInfo

# =========================
# Página
# =========================
st.title("🚚 Integração Distribuição")

# =========================
# Arquivos (pasta data/)
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"
ARQ_IDS = DATA_DIR / "Base IDs Logon.xlsx"
ARQ_RESPOSTAS = DATA_DIR / "Respostas Logon.xlsx"

# >>> NOVO: base de ADMITIDOS (ajuste o nome do arquivo se o seu for diferente)
ARQ_ADMITIDOS = DATA_DIR / "Admitidos.xlsx"

# =========================
# Última atualização dos dados (APENAS desta página)
# + Botão para forçar refresh de cache
# =========================
try:
    arquivos = [ARQ_ATIVOS, ARQ_IDS, ARQ_RESPOSTAS, ARQ_ADMITIDOS]
    last_mtime = max(a.stat().st_mtime for a in arquivos if a.exists())
    dt = datetime.fromtimestamp(last_mtime, tz=ZoneInfo("America/Sao_Paulo"))
    st.caption(f"🕒 Última atualização dos dados: {dt.strftime('%d/%m/%Y %H:%M')}")
except Exception:
    last_mtime = None
    st.caption("🕒 Última atualização: não disponível")

# Botão “mata-cache” (ajuda quando alguém troca arquivo e quer ver refletir na hora)
c_refresh, _ = st.columns([1, 5])
with c_refresh:
    if st.button("🔄 Atualizar dados agora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =========================
# Utils
# =========================
def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        unicodedata.normalize("NFKD", str(c).strip()).encode("ascii", "ignore").decode("utf-8").upper()
        for c in df.columns
    ]
    return df

def normalizar_texto(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s.upper().strip()

def limpar_texto(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def garantir_coluna(df: pd.DataFrame, col: str, default="") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df

def centralizar_tabela(df: pd.DataFrame):
    return (
        df.style
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )

def tratar_data_segura(series: pd.Series) -> pd.Series:
    """Aceita texto BR dd/mm/aaaa ou serial Excel (blindado)."""
    dt_txt = pd.to_datetime(series, errors="coerce", dayfirst=True)

    num = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = num.notna() & np.isfinite(num) & (num >= 20000) & (num <= 80000)  # ~1954 a ~2119

    dt_excel = pd.Series(pd.NaT, index=series.index)
    if mask.any():
        dt_excel.loc[mask] = pd.to_datetime(
            num.loc[mask].astype("int64"),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return dt_txt.fillna(dt_excel)

def preparar_excel_para_download(df: pd.DataFrame, sheet_name: str = "Dados") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()

def safe_filename(s: str) -> str:
    s = normalizar_texto(s)
    s = s.replace(" ", "_").replace("/", "-")
    return s

def opcoes(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    vals = (
        df[col].astype(str).str.strip()
        .replace(["", "nan", "None"], pd.NA)
        .dropna().unique().tolist()
    )
    return sorted(vals)

# =========================
# Status colors (coluna STATUS)
# =========================
STATUS_STYLE = {
    "Conforme esperado": "background-color: #22c55e; color: white; font-weight:700;",
    "Pendente mas no prazo": "background-color: #facc15; color: black; font-weight:700;",
    "Pendente em atraso": "background-color: #ef4444; color: white; font-weight:700;",
    "Concluido em atraso": "background-color: #fb923c; color: black; font-weight:700;",   # laranja
    "Concluido adiantado": "background-color: #fecaca; color: black; font-weight:700;",  # vermelho clarinho
}

# =========================
# Load (cache “inteligente” por mudança de arquivo)
# - quando você sobe/commita o Excel novo, muda o mtime -> invalida cache
# =========================
@st.cache_data(show_spinner=True)
def carregar_bases(_cache_buster: float | None):
    for arq in [ARQ_ATIVOS, ARQ_IDS, ARQ_RESPOSTAS, ARQ_ADMITIDOS]:
        if not arq.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {arq.name}")

    ativos_ = pd.read_excel(ARQ_ATIVOS)
    ids_ = pd.read_excel(ARQ_IDS)
    resp_ = pd.read_excel(ARQ_RESPOSTAS)
    admit_ = pd.read_excel(ARQ_ADMITIDOS)
    return ativos_, ids_, resp_, admit_

try:
    ativos, ids_logon, respostas, admitidos = carregar_bases(last_mtime)
except Exception as e:
    st.error(f"❌ Erro ao carregar arquivos da pasta /data: {e}")
    st.info(
        "✅ Confira se existem exatamente estes arquivos no GitHub na pasta /data (com .xlsx):\n"
        "- Base colaboradores ativos.xlsx\n"
        "- Base IDs Logon.xlsx\n"
        "- Respostas Logon.xlsx\n"
        "- Admitidos.xlsx"
    )
    st.stop()

# =========================
# Normalização
# =========================
ativos = normalizar_colunas(ativos)
ids_logon = normalizar_colunas(ids_logon)
respostas = normalizar_colunas(respostas)
admitidos = normalizar_colunas(admitidos)

# --- Ajustes de nomes de colunas (Admitidos) ---
# Esperado: COLABORADOR / DATA (admissão) / ATIVIDADE / OPERACAO / CARGO
for c in ["COLABORADOR", "DATA", "ATIVIDADE", "OPERACAO", "CARGO"]:
    admitidos = garantir_coluna(admitidos, c, "")

if "OPERAÇÃO" in admitidos.columns and "OPERACAO" not in admitidos.columns:
    admitidos["OPERACAO"] = admitidos["OPERAÇÃO"]

admitidos = limpar_texto(admitidos, ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE"])
admitidos["DATA_ADM_DT"] = tratar_data_segura(admitidos["DATA"])

# --- Ajustes de nomes de colunas (Ativos) ---
for c in ["COLABORADOR", "CARGO", "OPERACAO", "DATA ULT. ADM"]:
    ativos = garantir_coluna(ativos, c, "")

if "OPERAÇÃO" in ativos.columns and "OPERACAO" not in ativos.columns:
    ativos["OPERACAO"] = ativos["OPERAÇÃO"]

ativos = limpar_texto(ativos, ["COLABORADOR", "CARGO", "OPERACAO"])

# IDs logon
for c in ["COLABORADOR", "ID"]:
    ids_logon = garantir_coluna(ids_logon, c, "")
ids_logon = limpar_texto(ids_logon, ["COLABORADOR", "ID"])

# Respostas
for c in ["ID", "CURSO", "DATA ENTREGA"]:
    respostas = garantir_coluna(respostas, c, "")
respostas = limpar_texto(respostas, ["ID", "CURSO"])

# =========================
# Regras: cargos + data admissão
# (AGORA A BASE “FONTE” é ADMITIDOS, não a BASE ATIVOS)
# =========================
CARGOS_PERMITIDOS = [
    "Motorista Caminhão Distribuição",
    "Motorista de Distribuição AS",
    "Ajudante Distribuição",
    "Ajudante AS",
    "Motorista Entregador I",
    "Motorista Entregador",
    "Motorista Educador",
    "Motorista Entregador II",
]
CARGOS_PERMITIDOS_UP = [normalizar_texto(c) for c in CARGOS_PERMITIDOS]

admitidos["CARGO_UP"] = admitidos["CARGO"].astype(str).map(normalizar_texto)

limite = pd.to_datetime("2024-09-01")  # >= 01/09/2024

base = admitidos[
    (admitidos["CARGO_UP"].isin(CARGOS_PERMITIDOS_UP)) &
    (admitidos["DATA_ADM_DT"].notna()) &
    (admitidos["DATA_ADM_DT"] >= limite)
].copy()

# =========================
# Status do colaborador (Ativo/Inativo)
# - Ativo: aparece na Base colaboradores ativos
# - Inativo: não aparece
# =========================
ativos_set = set(ativos["COLABORADOR"].astype(str).map(normalizar_texto).replace({"": None}).dropna().tolist())
base["STATUS COLABORADOR"] = np.where(
    base["COLABORADOR"].astype(str).map(normalizar_texto).isin(ativos_set),
    "Ativo",
    "Inativo"
)

# --- EXCEÇÃO: CD PETRÓPOLIS com admissão em 16/07/2025 (ignorar) ---
data_excluir = pd.to_datetime("2025-07-16").date()
op_norm = base["OPERACAO"].astype(str).map(normalizar_texto)
adm_date = pd.to_datetime(base["DATA_ADM_DT"], errors="coerce").dt.date
base = base[~(op_norm.str.contains("CD PETROPOLIS", na=False) & (adm_date == data_excluir))]

# --- EXCLUSÕES POR NOME (ignorar no controle) ---
IGNORAR_NOMES = [
    "JULIANO CASTEDO MENDES",
    "CLEITON VINICIUS CESARIO DE CARVALHO",
    "ALEXANDER DE SOUZA GOMES",
]
ignorar_up = set(normalizar_texto(x) for x in IGNORAR_NOMES)
base = base[~base["COLABORADOR"].astype(str).map(normalizar_texto).isin(ignorar_up)]

# Datas derivadas
base["DATA ADMISSAO"] = pd.to_datetime(base["DATA_ADM_DT"], errors="coerce").dt.strftime("%d/%m/%Y").fillna("")
hoje = pd.Timestamp.now(tz=ZoneInfo("America/Sao_Paulo")).normalize().tz_localize(None)

# =========================
# IDs por colaborador
# =========================
ids_logon = ids_logon.drop_duplicates(subset=["COLABORADOR"], keep="last")

base = base.merge(ids_logon[["COLABORADOR", "ID"]], on="COLABORADOR", how="left")
base["ID"] = base["ID"].astype(str).str.strip().replace(["", "nan", "None"], pd.NA)

# =========================
# Respostas por ID + Curso (pega a primeira data)
# =========================
respostas["DATA_ENTREGA_DT"] = tratar_data_segura(respostas["DATA ENTREGA"])
respostas["ID"] = respostas["ID"].astype(str).str.strip()

resp_min = (
    respostas.dropna(subset=["ID"])
    .groupby(["ID", "CURSO"], dropna=False)["DATA_ENTREGA_DT"]
    .min()
    .reset_index()
)

# =========================
# Etapas + prazos
# (etapa, offset_min, offset_max)
# =========================
ETAPAS = [
    ("Dia 01 - Distribuição Urbana", 0, 3),
    ("Dia 02 - Distribuição Urbana", 1, 3),
    ("Dia 03 - Distribuição Urbana", 2, 4),
    ("Dia 04 - Distribuição Urbana", 3, 5),
    ("Dia 05 - Distribuição Urbana", 4, 7),
    ("Gradativa - Distribuição Urbana", 10, 14),
    ("1ª Quinzena - Distribuição Urbana", 12, 18),
    ("1° Mês - Distribuição Urbana", 24, 34),
]

# =========================
# Montagem da base LONGA (1 linha por etapa por colaborador)
# =========================
linhas = []
base_cols = ["COLABORADOR", "CARGO", "OPERACAO", "DATA ADMISSAO", "DATA_ADM_DT", "ID", "STATUS COLABORADOR", "ATIVIDADE"]

for _, r in base[base_cols].iterrows():
    adm = pd.to_datetime(r["DATA_ADM_DT"]).normalize()
    _id = r["ID"]

    for (etapa, off_min, off_max) in ETAPAS:
        prazo_min_dt = (adm + pd.Timedelta(days=off_min)).normalize()
        prazo_max_dt = (adm + pd.Timedelta(days=off_max)).normalize()

        realizado_dt = pd.NaT
        if pd.notna(_id):
            hit = resp_min[(resp_min["ID"] == str(_id)) & (resp_min["CURSO"] == etapa)]
            if len(hit) > 0:
                realizado_dt = hit["DATA_ENTREGA_DT"].iloc[0]

        if pd.notna(realizado_dt):
            realizado_dt = pd.to_datetime(realizado_dt).normalize()

        # DIAS (saldo do Prazo Máximo)
        if pd.isna(realizado_dt):
            dias = int((prazo_max_dt - hoje).days)  # pendente
            status = "Pendente em atraso" if hoje > prazo_max_dt else "Pendente mas no prazo"
            realizado_txt = ""
        else:
            dias = int((prazo_max_dt - realizado_dt).days)  # concluído (adiantado positivo, atraso negativo)
            if realizado_dt < prazo_min_dt:
                status = "Concluido adiantado"
            elif realizado_dt > prazo_max_dt:
                status = "Concluido em atraso"
            else:
                status = "Conforme esperado"
            realizado_txt = realizado_dt.strftime("%d/%m/%Y")

        linhas.append({
            "COLABORADOR": r["COLABORADOR"],
            "CARGO": r["CARGO"],
            "OPERACAO": r["OPERACAO"],
            "ATIVIDADE": r.get("ATIVIDADE", ""),
            "STATUS COLABORADOR": r["STATUS COLABORADOR"],
            "ADMISSAO": r["DATA ADMISSAO"],
            "ADMISSAO_DT": adm,
            "ETAPA": etapa,
            "PRAZO MINIMO": prazo_min_dt.strftime("%d/%m/%Y"),
            "PRAZO MAXIMO": prazo_max_dt.strftime("%d/%m/%Y"),
            "REALIZADO": realizado_txt,
            "DIAS": dias,
            "STATUS": status,
        })

etapas_df = pd.DataFrame(linhas)

# =========================
# Filtros (data em PT-BR)
# =========================
st.sidebar.header("Filtros")

min_adm = pd.to_datetime(etapas_df["ADMISSAO_DT"], errors="coerce").min()
max_adm = pd.to_datetime(etapas_df["ADMISSAO_DT"], errors="coerce").max()
if pd.isna(min_adm) or pd.isna(max_adm):
    min_adm = pd.to_datetime("2024-09-01")
    max_adm = pd.to_datetime(datetime.today().date()).normalize()

periodo = st.sidebar.date_input(
    "Período de admissão",
    value=(min_adm.date(), max_adm.date()),
    min_value=min_adm.date(),
    max_value=max_adm.date(),
    format="DD/MM/YYYY",  # <<< PT-BR no formato do teu dash
)
data_ini, data_fim = periodo

f_operacao = st.sidebar.multiselect("Operação", opcoes(etapas_df, "OPERACAO"))
f_cargo = st.sidebar.multiselect("Cargo", opcoes(etapas_df, "CARGO"))
f_etapa = st.sidebar.multiselect("Etapa", opcoes(etapas_df, "ETAPA"))
f_status = st.sidebar.multiselect(
    "Status (etapa)",
    ["Pendente em atraso", "Pendente mas no prazo", "Concluido em atraso", "Concluido adiantado", "Conforme esperado"]
)

# >>> NOVO: filtro por status do colaborador (Ativo/Inativo)
f_status_colab = st.sidebar.multiselect(
    "Status do colaborador",
    ["Ativo", "Inativo"],
    default=["Ativo", "Inativo"]
)

df_f = etapas_df.copy()
if f_operacao:
    df_f = df_f[df_f["OPERACAO"].isin(f_operacao)]
if f_cargo:
    df_f = df_f[df_f["CARGO"].isin(f_cargo)]
if f_etapa:
    df_f = df_f[df_f["ETAPA"].isin(f_etapa)]
if f_status:
    df_f = df_f[df_f["STATUS"].isin(f_status)]
if f_status_colab:
    df_f = df_f[df_f["STATUS COLABORADOR"].isin(f_status_colab)]

df_f = df_f[
    (pd.to_datetime(df_f["ADMISSAO_DT"]).dt.date >= data_ini) &
    (pd.to_datetime(df_f["ADMISSAO_DT"]).dt.date <= data_fim)
]

# =========================
# Ordenação geral por ADMISSAO (crescente)
# =========================
df_f["_ADM_DT"] = pd.to_datetime(df_f["ADMISSAO_DT"], errors="coerce")
df_f = df_f.sort_values(["_ADM_DT", "COLABORADOR", "ETAPA"]).drop(columns=["_ADM_DT"])

# =========================
# Cards
# =========================
total_linhas = len(df_f)
pend_atraso = int((df_f["STATUS"] == "Pendente em atraso").sum())
pend_prazo = int((df_f["STATUS"] == "Pendente mas no prazo").sum())
conc_ok = int((df_f["STATUS"] == "Conforme esperado").sum())
conc_adiant = int((df_f["STATUS"] == "Concluido adiantado").sum())

no_prazo_vencendo_3 = int(
    ((df_f["STATUS"] == "Pendente mas no prazo") & (df_f["DIAS"] >= 0) & (df_f["DIAS"] <= 3)).sum()
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Etapas (linhas)", total_linhas)
c2.metric("🔴 Pendente em atraso", pend_atraso)
c3.metric("🟡 Pendente no prazo", pend_prazo)
c4.metric("🟡 No prazo vencendo em até 3 dias", no_prazo_vencendo_3)
c5.metric("🟢 Conforme esperado", conc_ok)
c6.metric("⚡ Concluído adiantado", conc_adiant)

st.divider()

# =========================
# 1 - 📌 Aderência Média - Log20
# OK = Conforme esperado + Pendente mas no prazo
# =========================
st.subheader("📌 Aderência Média - Log20")

total_etapas = len(df_f)
total_ok = int(((df_f["STATUS"] == "Conforme esperado") | (df_f["STATUS"] == "Pendente mas no prazo")).sum())
aderencia_total = (total_ok / total_etapas) if total_etapas > 0 else 0.0
pct = aderencia_total * 100

if pct >= 100:
    cor = "#22c55e"  # verde
elif pct >= 80:
    cor = "#facc15"  # amarelo
else:
    cor = "#ef4444"  # vermelho

col_bar, col_pct = st.columns([8, 1])
with col_bar:
    st.markdown(
        f"""
        <div style="width: 100%; background: #2a2f3a; border-radius: 999px; height: 12px; overflow: hidden;">
          <div style="width: {min(pct, 100):.2f}%; background: {cor}; height: 12px;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_pct:
    st.write(f"**{pct:.2f}%**")

st.divider()

# =========================
# 2 - Aderência por Operação
# =========================
st.subheader("📊 Aderência por Operação")

ader_oper = (
    df_f.groupby("OPERACAO", dropna=False)
    .agg(
        TOTAL=("STATUS", "size"),
        OK=("STATUS", lambda s: ((s == "Conforme esperado") | (s == "Pendente mas no prazo")).sum()),
    )
    .reset_index()
)

ader_oper["ADERENCIA_%"] = np.where(
    ader_oper["TOTAL"] > 0,
    (ader_oper["OK"] / ader_oper["TOTAL"]) * 100,
    0
)

ader_oper = ader_oper.sort_values("ADERENCIA_%", ascending=False)

if len(ader_oper) == 0:
    st.info("Sem dados para calcular aderência com os filtros atuais.")
else:
    fig_ad = px.bar(
        ader_oper,
        x="OPERACAO",
        y="ADERENCIA_%",
        text=ader_oper["ADERENCIA_%"].map(lambda x: f"{x:.2f}%"),
    )
    fig_ad.update_layout(yaxis_title="Aderência (%)", xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig_ad, use_container_width=True)

st.divider()

# =========================
# 3 - 🔴 Pendentes em atraso (tabela) - ordem por ADMISSAO crescente
# =========================
st.subheader("🔴 Pendentes em atraso")

atraso_df = df_f[df_f["STATUS"] == "Pendente em atraso"].copy()
cols_atraso = ["COLABORADOR", "STATUS COLABORADOR", "CARGO", "OPERACAO", "ADMISSAO", "ETAPA", "PRAZO MAXIMO", "DIAS"]
atraso_df = atraso_df[[c for c in cols_atraso if c in atraso_df.columns]]

if len(atraso_df) == 0:
    st.info("Nenhuma etapa em atraso com os filtros atuais.")
else:
    atraso_df["_ADM_DT"] = pd.to_datetime(atraso_df["ADMISSAO"], dayfirst=True, errors="coerce")
    atraso_df = atraso_df.sort_values(["_ADM_DT", "COLABORADOR", "ETAPA"]).drop(columns=["_ADM_DT"])
    st.dataframe(centralizar_tabela(atraso_df), use_container_width=True)

st.divider()

# =========================
# 4 - 🟡 No prazo vencendo em até 3 dias (tabela) - ordem por ADMISSAO crescente
# =========================
vencendo_3_df = df_f[
    (df_f["STATUS"] == "Pendente mas no prazo") &
    (df_f["DIAS"] >= 0) &
    (df_f["DIAS"] <= 3)
].copy()

cols_alerta = ["COLABORADOR", "STATUS COLABORADOR", "CARGO", "OPERACAO", "ADMISSAO", "ETAPA", "PRAZO MAXIMO", "DIAS"]
vencendo_3_df = vencendo_3_df[[c for c in cols_alerta if c in vencendo_3_df.columns]]

st.subheader("🟡 No prazo vencendo em até 3 dias")
if len(vencendo_3_df) == 0:
    st.info("Nenhuma etapa 'Pendente mas no prazo' vencendo em até 3 dias com os filtros atuais.")
else:
    vencendo_3_df["_ADM_DT"] = pd.to_datetime(vencendo_3_df["ADMISSAO"], dayfirst=True, errors="coerce")
    vencendo_3_df = vencendo_3_df.sort_values(["_ADM_DT", "COLABORADOR", "DIAS"]).drop(columns=["_ADM_DT"])
    st.dataframe(centralizar_tabela(vencendo_3_df), use_container_width=True)

st.divider()

# =========================
# 5 - 📋 Acompanhamento por Etapa
# =========================
st.subheader("📋 Acompanhamento por Etapa")

ordem_etapas = [e[0] for e in ETAPAS]
tabs = st.tabs(ordem_etapas)

cols_ordem = [
    "COLABORADOR",
    "STATUS COLABORADOR",  # <<< NOVO
    "CARGO",
    "OPERACAO",
    "ATIVIDADE",
    "ADMISSAO",
    "ETAPA",
    "PRAZO MINIMO",
    "PRAZO MAXIMO",
    "REALIZADO",
    "DIAS",
    "STATUS",
]

for i, etapa in enumerate(ordem_etapas):
    with tabs[i]:
        df_etapa = df_f[df_f["ETAPA"] == etapa].copy()

        if len(df_etapa) == 0:
            st.info("Sem dados para esta etapa com os filtros atuais.")
            continue

        df_etapa["_ADM_DT"] = pd.to_datetime(df_etapa["ADMISSAO_DT"], errors="coerce")
        df_etapa = df_etapa.sort_values(["_ADM_DT", "COLABORADOR"]).drop(columns=["_ADM_DT"])
        df_etapa = df_etapa[[c for c in cols_ordem if c in df_etapa.columns]]

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Registros", len(df_etapa))
        cc2.metric("🔴 Pendente em atraso", int((df_etapa["STATUS"] == "Pendente em atraso").sum()))
        cc3.metric("🟡 Pendente no prazo", int((df_etapa["STATUS"] == "Pendente mas no prazo").sum()))

        def style_cell(col_name: str, row: pd.Series) -> str:
            if col_name == "STATUS":
                return STATUS_STYLE.get(str(row["STATUS"]), "")
            if col_name == "DIAS" and str(row["STATUS"]) == "Pendente em atraso":
                return "color: red; font-weight:800;"
            return ""

        sty = centralizar_tabela(df_etapa).apply(
            lambda col: [style_cell(col.name, r) for _, r in df_etapa.iterrows()],
            axis=0
        )

        st.dataframe(sty, use_container_width=True)

        excel_etapa = preparar_excel_para_download(df_etapa, sheet_name="Etapa")
        st.download_button(
            label=f"⬇️ Baixar Excel ({etapa})",
            data=excel_etapa,
            file_name=f"integracao_{safe_filename(etapa)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

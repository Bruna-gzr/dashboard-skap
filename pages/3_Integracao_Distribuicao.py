import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from io import BytesIO
import unicodedata
import plotly.express as px

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

# =========================
# Utils
# =========================
def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        unicodedata.normalize("NFKD", str(c).strip()).encode("ascii", "ignore").decode("utf-8").upper()
        for c in df.columns
    ]
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

# =========================
# Load
# =========================
@st.cache_data(show_spinner=True)
def carregar_bases():
    for arq in [ARQ_ATIVOS, ARQ_IDS, ARQ_RESPOSTAS]:
        if not arq.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {arq}")

    ativos = pd.read_excel(ARQ_ATIVOS)
    ids = pd.read_excel(ARQ_IDS)
    resp = pd.read_excel(ARQ_RESPOSTAS)
    return ativos, ids, resp

try:
    ativos, ids_logon, respostas = carregar_bases()
except Exception as e:
    st.error(f"❌ Erro ao carregar arquivos da pasta /data: {e}")
    st.info(
        "✅ Suba estes arquivos no GitHub na pasta /data (com .xlsx):\n"
        "- Base colaboradores ativos.xlsx\n"
        "- Base IDs Logon.xlsx\n"
        "- Respostas Logon.xlsx"
    )
    st.stop()

# =========================
# Normalização
# =========================
ativos = normalizar_colunas(ativos)
ids_logon = normalizar_colunas(ids_logon)
respostas = normalizar_colunas(respostas)

# Colunas mínimas (ativos)
for c in ["COLABORADOR", "CARGO", "OPERACAO", "DATA ULT. ADM"]:
    ativos = garantir_coluna(ativos, c, "")

# Caso venha com acento "OPERAÇÃO"
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
CARGOS_PERMITIDOS_UP = [c.strip().upper() for c in CARGOS_PERMITIDOS]

ativos["CARGO_UP"] = ativos["CARGO"].astype(str).str.strip().str.upper()

ativos["DATA_ADM_DT"] = tratar_data_segura(ativos["DATA ULT. ADM"])
limite = pd.to_datetime("2024-09-01")  # >= 01/09/2024

base = ativos[
    (ativos["CARGO_UP"].isin(CARGOS_PERMITIDOS_UP)) &
    (ativos["DATA_ADM_DT"].notna()) &
    (ativos["DATA_ADM_DT"] >= limite)
].copy()

# --- EXCEÇÃO: CD PETRÓPOLIS com admissão em 16/07/2025 (ignorar) ---
data_excluir = pd.to_datetime("2025-07-16").date()

def normalizar_texto(s: str) -> str:
    import unicodedata
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s.upper().strip()

op_norm = base["OPERACAO"].astype(str).map(normalizar_texto)
adm_date = base["DATA_ADM_DT"].dt.date  # tira hora, compara só a data

base = base[~((op_norm.str.contains("CD PETROPOLIS", na=False)) & (adm_date == data_excluir))]


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
# =========================
ETAPAS = [
    ("Dia 01 - Distribuição Urbana", 0, 3, 3),
    ("Dia 02 - Distribuição Urbana", 1, 3, 2),
    ("Dia 03 - Distribuição Urbana", 2, 4, 2),
    ("Dia 04 - Distribuição Urbana", 3, 5, 2),
    ("Dia 05 - Distribuição Urbana", 4, 7, 3),
    ("Gradativa - Distribuição Urbana", 10, 14, 3),
    ("1ª Quinzena - Distribuição Urbana", 12, 18, 6),
    ("1° Mês - Distribuição Urbana", 24, 34, 10),
]

def calcular_status(realizado_dt: pd.Timestamp, dias: int, lim_adiantado: int) -> str:
    if pd.isna(realizado_dt):
        if dias < 0:
            return "Pendente em atraso"
        return "Pendente mas no prazo"
    if dias < 0:
        return "Concluido em atraso"
    if dias > lim_adiantado:
        return "Concluido adiantado"
    return "Conforme esperado"

# =========================
# Montagem da base LONGA (1 linha por etapa por colaborador)
# =========================
linhas = []
base_cols = ["COLABORADOR", "CARGO", "OPERACAO", "DATA ADMISSAO", "DATA_ADM_DT", "ID"]

for _, r in base[base_cols].iterrows():
    adm = r["DATA_ADM_DT"]
    _id = r["ID"]

    for (etapa, off_min, off_max, lim_adiantado) in ETAPAS:
        prazo_min_dt = adm + pd.Timedelta(days=off_min)
        prazo_max_dt = adm + pd.Timedelta(days=off_max)

        realizado_dt = pd.NaT
        if pd.notna(_id):
            hit = resp_min[(resp_min["ID"] == str(_id)) & (resp_min["CURSO"] == etapa)]
            if len(hit) > 0:
                realizado_dt = hit["DATA_ENTREGA_DT"].iloc[0]

        if pd.isna(realizado_dt):
            dias = int((prazo_max_dt - hoje).days)
        else:
            dias = int((prazo_max_dt - realizado_dt).days)

        status = calcular_status(realizado_dt, dias, lim_adiantado)

        linhas.append({
    "COLABORADOR": r["COLABORADOR"],
    "CARGO": r["CARGO"],
    "OPERACAO": r["OPERACAO"],
    "ADMISSAO": r["DATA ADMISSAO"],  # ✅ NOVO
    "ETAPA": etapa,
    "PRAZO MINIMO": prazo_min_dt.strftime("%d/%m/%Y"),
    "PRAZO MAXIMO": prazo_max_dt.strftime("%d/%m/%Y"),
    "REALIZADO": "" if pd.isna(realizado_dt) else realizado_dt.strftime("%d/%m/%Y"),
    "DIAS": dias,
    "STATUS": status,
})

etapas_df = pd.DataFrame(linhas)

# =========================
# Filtros
# =========================
st.sidebar.header("Filtros")

def opcoes(df, col):
    if col not in df.columns:
        return []
    vals = (
        df[col].astype(str).str.strip()
        .replace(["", "nan", "None"], pd.NA)
        .dropna().unique().tolist()
    )
    return sorted(vals)

f_operacao = st.sidebar.multiselect("Operação", opcoes(etapas_df, "OPERACAO"))
f_cargo = st.sidebar.multiselect("Cargo", opcoes(etapas_df, "CARGO"))
f_etapa = st.sidebar.multiselect("Etapa", opcoes(etapas_df, "ETAPA"))
f_status = st.sidebar.multiselect(
    "Status",
    ["Pendente em atraso", "Pendente mas no prazo", "Concluido em atraso", "Concluido adiantado", "Conforme esperado"]
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

# =========================
# Cards
# =========================
total_linhas = len(df_f)
pend_atraso = int((df_f["STATUS"] == "Pendente em atraso").sum())
pend_prazo = int((df_f["STATUS"] == "Pendente mas no prazo").sum())
conc_atraso = int((df_f["STATUS"] == "Concluido em atraso").sum())
conc_ok = int((df_f["STATUS"] == "Conforme esperado").sum())
conc_adiant = int((df_f["STATUS"] == "Concluido adiantado").sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Linhas (etapas)", total_linhas)
c2.metric("🔴 Pendente em atraso", pend_atraso)
c3.metric("🟡 Pendente no prazo", pend_prazo)
c4.metric("🟢 Conforme esperado", conc_ok)
c5.metric("⚡ Adiantado", conc_adiant)

st.divider()

# =========================
# Gráfico: Pendentes em atraso por Operação
# =========================
st.subheader("🔴 Pendentes em atraso por Operação")
g = (
    df_f[df_f["STATUS"] == "Pendente em atraso"]
    .groupby("OPERACAO", dropna=False)
    .size()
    .sort_values(ascending=False)
    .reset_index(name="Quantidade")
)

if len(g) == 0:
    st.info("Sem pendências em atraso com os filtros atuais.")
else:
    fig = px.bar(g, x="OPERACAO", y="Quantidade", text="Quantidade")
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# Tabela principal
# =========================
st.subheader("📋 Acompanhamento por Curso")

# Ordem fixa das etapas
ordem_etapas = [e[0] for e in ETAPAS]

# Tabs: (opcional) primeira aba visão geral + uma para cada etapa
tab_labels = ["📌 Visão Geral"] + ordem_etapas
tabs = st.tabs(tab_labels)

# -------------------------
# Aba 0: Visão Geral
# -------------------------
with tabs[0]:
    st.write("Resumo do acompanhamento com os filtros atuais.")

    # Cards rápidos por status (no contexto filtrado)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Etapas", len(df_f))
    c2.metric("🔴 Pendente em atraso", int((df_f["STATUS"] == "Pendente em atraso").sum()))
    c3.metric("🟡 Pendente no prazo", int((df_f["STATUS"] == "Pendente mas no prazo").sum()))
    c4.metric("🟢 Conforme esperado", int((df_f["STATUS"] == "Conforme esperado").sum()))
    c5.metric("⚡ Concluído adiantado", int((df_f["STATUS"] == "Concluido adiantado").sum()))

    st.divider()

    # Lista geral (todas as etapas)
    df_view = df_f.copy()
    df_view["ETAPA"] = pd.Categorical(df_view["ETAPA"], categories=ordem_etapas, ordered=True)
    df_view = df_view.sort_values(["OPERACAO", "COLABORADOR", "ETAPA"])

    if len(df_view) == 0:
        st.info("Sem dados para exibir com os filtros atuais.")
    else:
        # DIAS em vermelho apenas quando pendente em atraso
        def colorir_dias(row):
            return "color: red; font-weight:700;" if row["STATUS"] == "Pendente em atraso" else ""

        sty = centralizar_tabela(df_view).apply(
            lambda col: [colorir_dias(r) if col.name == "DIAS" else "" for _, r in df_view.iterrows()],
            axis=0
        )
        st.dataframe(sty, use_container_width=True)

        excel_bytes = preparar_excel_para_download(df_view, sheet_name="Visao_Geral")
        st.download_button(
            label="⬇️ Baixar Excel (Visão Geral)",
            data=excel_bytes,
            file_name="integracao_distribuicao_visao_geral.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# -------------------------
# Abas por etapa/curso
# -------------------------
for i, etapa in enumerate(ordem_etapas, start=1):
    with tabs[i]:
        st.write(f"Etapa: **{etapa}**")

        df_etapa = df_f[df_f["ETAPA"] == etapa].copy()
        # ordenar por admissão e reorganizar colunas
df_etapa["_ADM_DT"] = pd.to_datetime(df_etapa["ADMISSAO"], dayfirst=True, errors="coerce")
df_etapa = df_etapa.sort_values(["OPERACAO", "_ADM_DT", "COLABORADOR"]).drop(columns=["_ADM_DT"])

cols_ordem = ["COLABORADOR", "CARGO", "OPERACAO", "ADMISSAO", "ETAPA", "PRAZO MINIMO", "PRAZO MAXIMO", "REALIZADO", "DIAS", "STATUS"]
df_etapa = df_etapa[[c for c in cols_ordem if c in df_etapa.columns]]

        df_etapa = df_etapa.sort_values(["OPERACAO", "COLABORADOR"])

        if len(df_etapa) == 0:
            st.info("Sem dados para esta etapa com os filtros atuais.")
        else:
            # cards por etapa
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Registros", len(df_etapa))
            cc2.metric("🔴 Pendente em atraso", int((df_etapa["STATUS"] == "Pendente em atraso").sum()))
            cc3.metric("🟡 Pendente no prazo", int((df_etapa["STATUS"] == "Pendente mas no prazo").sum()))

            # DIAS em vermelho apenas quando pendente em atraso
            def colorir_dias(row):
                return "color: red; font-weight:700;" if row["STATUS"] == "Pendente em atraso" else ""

            sty = centralizar_tabela(df_etapa).apply(
                lambda col: [colorir_dias(r) if col.name == "DIAS" else "" for _, r in df_etapa.iterrows()],
                axis=0
            )
            st.dataframe(sty, use_container_width=True)

            # Exportação logo abaixo (por etapa)
            excel_etapa = preparar_excel_para_download(df_etapa, sheet_name=etapa)
            nome_arquivo = etapa.replace("/", "-").replace("ª", "a").replace("°", "o").replace(" ", "_")
            st.download_button(
                label=f"⬇️ Baixar Excel ({etapa})",
                data=excel_etapa,
                file_name=f"integracao_{nome_arquivo}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


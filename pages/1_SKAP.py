import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
import numpy as np
from io import BytesIO

# =========================
# Config
# =========================
st.title("📊 Painel SKAP")

# =========================
# Carregamento automático (pasta data/)
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARQ_SKAP = DATA_DIR / "Skap.xlsx"
ARQ_COM = DATA_DIR / "Skap - comentarios.xlsx"

@st.cache_data(show_spinner=True)
def carregar_dados():
    if not ARQ_SKAP.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_SKAP}")
    if not ARQ_COM.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_COM}")

    skap_df = pd.read_excel(ARQ_SKAP)
    com_df = pd.read_excel(ARQ_COM)
    return skap_df, com_df

try:
    skap, comentarios = carregar_dados()
except Exception as e:
    st.error(f"❌ Erro ao carregar os arquivos da pasta /data: {e}")
    st.info("✅ Verifique se existem exatamente estes arquivos no GitHub: data/Skap.xlsx e data/Skap - comentarios.xlsx")
    st.stop()

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

def centralizar_tabela(df: pd.DataFrame):
    return (
        df.style
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )


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
    """Consolida colunas duplicadas do merge: NOME_X/NOME_Y -> NOME."""
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
    """
    Converte data em:
    - aceita texto BR dd/mm/aaaa
    - aceita serial excel (somente faixa plausível e finita)
    Evita overflow no Streamlit Cloud.
    """
    df = garantir_coluna(df, col_data, "")
    df[f"{col_data}_RAW"] = df[col_data]

    # 1) tenta parse como texto (BR)
    dt_txt = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)

    # 2) serial do Excel (blindado)
    num = pd.to_numeric(df[col_data], errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan)

    mask = num.notna() & np.isfinite(num) & (num >= 20000) & (num <= 80000)  # ~1954 a ~2119

    dt_excel = pd.Series(pd.NaT, index=df.index)
    if mask.any():
        dt_excel.loc[mask] = pd.to_datetime(
            num.loc[mask].astype("int64"),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

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

def percent_str(x) -> str:
    try:
        return f"{float(x):.0%}"
    except Exception:
        return "0%"

# =========================
# Normalização + colunas mínimas
# =========================
skap = normalizar_colunas(skap)
comentarios = normalizar_colunas(comentarios)

# garante colunas mínimas no SKAP
for col in [
    "COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA", "DATA ULT. ADM",
    "HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS", "HABILIDADES EMPODERAMENTO",
    "NIVEIS"
]:
    skap = garantir_coluna(skap, col, "")

skap = limpar_texto(skap, ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA", "NIVEIS"])
comentarios = limpar_texto(comentarios, ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA"])

# =========================
# Datas + métricas
# =========================
skap = tratar_data_adm(skap, "DATA ULT. ADM")

# Percentuais (como número 0..1)
for col in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS", "HABILIDADES EMPODERAMENTO"]:
    skap = garantir_coluna(skap, col, 0)
    skap[col] = pd.to_numeric(skap[col], errors="coerce").fillna(0)

# Prazos (data real + versão texto)
skap["PRAZO_TECNICAS_DT"] = skap["DATA ULT. ADM"] + pd.Timedelta(days=30)
skap["PRAZO_ESPECIFICAS_DT"] = skap["DATA ULT. ADM"] + pd.Timedelta(days=60)

skap["PRAZO TECNICAS"] = skap["PRAZO_TECNICAS_DT"].dt.strftime("%d/%m/%Y").fillna("")
skap["PRAZO ESPECIFICAS"] = skap["PRAZO_ESPECIFICAS_DT"].dt.strftime("%d/%m/%Y").fillna("")

# Status (suas regras)
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

for c in ["CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA", "NIVEIS"]:
    base = consolidar_xy(base, c)

# =========================
# Ordem das colunas (EXATA + ajustes pedidos)
# - Inserir HABILIDADES EMPODERAMENTO após STATUS ESPECIFICAS
# - Inserir NIVEIS ao lado (logo após)
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
    "HABILIDADES EMPODERAMENTO",
    "NIVEIS",
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
f_niveis = st.sidebar.multiselect("Níveis", opcoes(base, "NIVEIS"))
f_status = st.sidebar.multiselect("Status", ["Realizado", "Não realizado", "No prazo"])

base_f = base.copy()
if f_operacao:
    base_f = base_f[base_f["OPERACAO"].isin(f_operacao)]
if f_lideranca:
    base_f = base_f[base_f["LIDERANCA"].isin(f_lideranca)]
if f_atividade:
    base_f = base_f[base_f["ATIVIDADE"].isin(f_atividade)]
if f_niveis:
    base_f = base_f[base_f["NIVEIS"].isin(f_niveis)]
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
# Gráficos (ordem decrescente)
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

if len(gop) == 0:
    st.info("Sem pendências no filtro atual.")
else:
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

if len(glid) == 0:
    st.info("Sem pendências no filtro atual.")
else:
    fig2 = px.bar(glid, x="LIDERANCA", y="Quantidade", text="Quantidade")
    fig2.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# Atenção: Vencimento próximo (até 7 dias)
# - Apenas etapas NÃO realizadas
# =========================
hoje = pd.to_datetime(datetime.today().date())

# reconstruir datas de prazo a partir da base original (skap) com join por colaborador
# (usamos as datas calculadas no skap para não depender de strings)
tmp = skap[[
    "COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA",
    "HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS",
    "PRAZO_TECNICAS_DT", "PRAZO_ESPECIFICAS_DT",
    "STATUS TECNICAS", "STATUS ESPECIFICAS"
]].copy()

# aplica os mesmos filtros na tmp (para respeitar o que usuário filtrou)
tmp = normalizar_colunas(tmp)
tmp = tmp.merge(
    base_f[["COLABORADOR"]].drop_duplicates(),
    on="COLABORADOR",
    how="inner"
)

alertas = []

# Técnicas vencendo
mask_tec = (tmp["STATUS TECNICAS"] != "Realizado") & tmp["PRAZO_TECNICAS_DT"].notna()
dias_tec = (tmp.loc[mask_tec, "PRAZO_TECNICAS_DT"] - hoje).dt.days
tmp_tec = tmp.loc[mask_tec].copy()
tmp_tec["DIAS PARA VENCER"] = dias_tec.values
tmp_tec = tmp_tec[(tmp_tec["DIAS PARA VENCER"] >= 0) & (tmp_tec["DIAS PARA VENCER"] <= 7)]

if len(tmp_tec) > 0:
    tmp_tec["ETAPA"] = "Técnicas"
    tmp_tec["DATA VENCIMENTO"] = tmp_tec["PRAZO_TECNICAS_DT"].dt.strftime("%d/%m/%Y")
    alertas.append(tmp_tec[["COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA", "ETAPA", "DATA VENCIMENTO", "DIAS PARA VENCER"]])

# Específicas vencendo
mask_esp = (tmp["STATUS ESPECIFICAS"] != "Realizado") & tmp["PRAZO_ESPECIFICAS_DT"].notna()
dias_esp = (tmp.loc[mask_esp, "PRAZO_ESPECIFICAS_DT"] - hoje).dt.days
tmp_esp = tmp.loc[mask_esp].copy()
tmp_esp["DIAS PARA VENCER"] = dias_esp.values
tmp_esp = tmp_esp[(tmp_esp["DIAS PARA VENCER"] >= 0) & (tmp_esp["DIAS PARA VENCER"] <= 7)]

if len(tmp_esp) > 0:
    tmp_esp["ETAPA"] = "Específicas"
    tmp_esp["DATA VENCIMENTO"] = tmp_esp["PRAZO_ESPECIFICAS_DT"].dt.strftime("%d/%m/%Y")
    alertas.append(tmp_esp[["COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA", "ETAPA", "DATA VENCIMENTO", "DIAS PARA VENCER"]])

alerta_df = pd.concat(alertas, ignore_index=True) if alertas else pd.DataFrame(
    columns=["COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA", "ETAPA", "DATA VENCIMENTO", "DIAS PARA VENCER"]
)

if len(alerta_df) > 0:
    alerta_df = alerta_df.sort_values(["DIAS PARA VENCER", "OPERACAO", "LIDERANCA", "COLABORADOR"], ascending=[True, True, True, True])

st.subheader("⚠️ Atenção: Vencimento próximo")
if len(alerta_df) == 0:
    st.info("Nenhuma etapa vencendo nos próximos 7 dias com os filtros atuais.")
else:
    st.dataframe(centralizar_tabela(alerta_df), use_container_width=True)

st.divider()

# =========================
# Exportar pendentes (filtrados)
# - Pendentes = qualquer etapa que NÃO esteja Realizado (No prazo ou Não realizado)
# =========================
pendentes_df = base_f[
    (base_f["STATUS TECNICAS"] != "Realizado") |
    (base_f["STATUS ESPECIFICAS"] != "Realizado")
].copy()

def preparar_excel_para_download(df: pd.DataFrame) -> bytes:
    export_df = df.copy()

    # formatar percentuais como texto no arquivo exportado
    for c in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS", "HABILIDADES EMPODERAMENTO"]:
        if c in export_df.columns:
            export_df[c] = pd.to_numeric(export_df[c], errors="coerce").fillna(0).map(lambda x: f"{x:.0%}")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Pendentes")
    return output.getvalue()

st.subheader("📤 Exportação")
col_a, col_b = st.columns([1, 2])
with col_a:
    st.write(f"Pendentes no filtro atual: **{len(pendentes_df)}**")

if len(pendentes_df) == 0:
    st.info("Sem pendentes para exportar com os filtros atuais.")
else:
    excel_bytes = preparar_excel_para_download(pendentes_df)
    st.download_button(
        label="⬇️ Baixar Excel (pendentes filtrados)",
        data=excel_bytes,
        file_name="pendentes_filtrados_skap.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.divider()

# =========================
# Tabela principal (sem Styler)
# =========================
st.subheader("📋 Detalhamento Individual")

tabela = base_f.copy()

# Percentuais como texto (97%)
for c in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS", "HABILIDADES EMPODERAMENTO"]:
    if c in tabela.columns:
        tabela[c] = pd.to_numeric(tabela[c], errors="coerce").fillna(0).map(lambda x: f"{x:.0%}")

# Status com emoji (visual e simples)
def emoji_status(s):
    if s == "Não realizado":
        return "🔴 Não realizado"
    if s == "Realizado":
        return "🟢 Realizado"
    if s == "No prazo":
        return "🟡 No prazo"
    return s

for c in ["STATUS TECNICAS", "STATUS ESPECIFICAS"]:
    if c in tabela.columns:
        tabela[c] = tabela[c].astype(str).map(emoji_status)

st.dataframe(centralizar_tabela(tabela), use_container_width=True)




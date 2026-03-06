import streamlit as st

st.set_page_config(
    page_title="Dashboard RH",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
import numpy as np
from io import BytesIO
from zoneinfo import ZoneInfo
import unicodedata

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
ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"

# =========================
# Última atualização dos dados + cache que invalida quando arquivo muda
# + botão para forçar atualização
# =========================
def get_last_mtime():
    arquivos = [ARQ_SKAP, ARQ_COM, ARQ_ATIVOS]
    return max(a.stat().st_mtime for a in arquivos if a.exists())

@st.cache_data(show_spinner=True)
def carregar_dados(_cache_key: float):
    if not ARQ_SKAP.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_SKAP}")
    if not ARQ_COM.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_COM}")
    if not ARQ_ATIVOS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_ATIVOS}")

    skap_df = pd.read_excel(ARQ_SKAP)
    com_df = pd.read_excel(ARQ_COM)
    ativos_df = pd.read_excel(ARQ_ATIVOS)

    return skap_df, com_df, ativos_df

try:
    last_mtime = get_last_mtime()
    dt = datetime.fromtimestamp(last_mtime, tz=ZoneInfo("America/Sao_Paulo"))
    st.caption(f"🕒 Última atualização dos dados: {dt.strftime('%d/%m/%Y %H:%M')}")

    c_refresh, _ = st.columns([1, 5])
    with c_refresh:
        if st.button("🔄 Atualizar dados agora", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    skap, comentarios, ativos = carregar_dados(last_mtime)

except Exception as e:
    st.error(f"❌ Erro ao carregar os arquivos da pasta /data: {e}")
    st.info(
        "✅ Verifique se existem exatamente estes arquivos no GitHub: "
        "data/Skap.xlsx, data/Skap - comentarios.xlsx e data/Base colaboradores ativos.xlsx"
    )
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

def normalizar_texto(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFKD", s).encode("ascii", errors="ignore").decode("utf-8")
    s = " ".join(s.split())
    return s

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

def tratar_data_adm(df: pd.DataFrame, col_data: str = "DATA ULT. ADM") -> pd.DataFrame:
    """
    Converte data em:
    - aceita texto BR dd/mm/aaaa
    - aceita serial excel (somente faixa plausível e finita)
    Evita overflow no Streamlit Cloud.
    """
    df = garantir_coluna(df, col_data, "")
    df[f"{col_data}_RAW"] = df[col_data]

    dt_txt = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)

    num = pd.to_numeric(df[col_data], errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan)
    mask = num.notna() & np.isfinite(num) & (num >= 20000) & (num <= 80000)

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

def preparar_excel_para_download(df: pd.DataFrame, sheet_name: str = "Dados") -> bytes:
    export_df = df.copy()
    for c in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS", "HABILIDADES EMPODERAMENTO"]:
        if c in export_df.columns:
            export_df[c] = pd.to_numeric(export_df[c], errors="coerce").fillna(0).map(lambda x: f"{x:.0%}")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()

# =========================
# Normalização
# =========================
skap = normalizar_colunas(skap)
comentarios = normalizar_colunas(comentarios)
ativos = normalizar_colunas(ativos)

# =========================
# Garantir colunas mínimas
# =========================
for col in [
    "COLABORADOR",
    "HABILIDADES TECNICAS",
    "HABILIDADES ESPECIFICAS",
    "HABILIDADES EMPODERAMENTO",
    "NIVEIS"
]:
    skap = garantir_coluna(skap, col, "")

for col in ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA", "DATA ULT. ADM"]:
    ativos = garantir_coluna(ativos, col, "")

for col in ["COLABORADOR"]:
    comentarios = garantir_coluna(comentarios, col, "")

skap = limpar_texto(skap, ["COLABORADOR", "NIVEIS"])
comentarios = limpar_texto(comentarios, ["COLABORADOR"])
ativos = limpar_texto(ativos, ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "LIDERANCA"])

# garante colunas de comentário se existirem
for col in ["HABILIDADE TECNICA", "HABILIDADE ESPECIFICA", "HABILIDADE EMPODERAMENTO"]:
    comentarios = garantir_coluna(comentarios, col, "")

# =========================
# Chave normalizada para cruzar por nome
# =========================
skap["CHAVE_COLABORADOR"] = skap["COLABORADOR"].apply(normalizar_texto)
comentarios["CHAVE_COLABORADOR"] = comentarios["COLABORADOR"].apply(normalizar_texto)
ativos["CHAVE_COLABORADOR"] = ativos["COLABORADOR"].apply(normalizar_texto)

# Mantém 1 linha por colaborador na base de ativos
ativos = ativos.drop_duplicates(subset=["CHAVE_COLABORADOR"], keep="first").copy()

# =========================
# Merge SKAP + comentários
# =========================
base = skap.merge(
    comentarios.drop(columns=["COLABORADOR"], errors="ignore"),
    on="CHAVE_COLABORADOR",
    how="left",
    suffixes=("", "_COM")
).fillna("")

# =========================
# Trazer dados OFICIAIS da base de ativos
# =========================
ativos_ref = ativos[
    [
        "CHAVE_COLABORADOR",
        "CARGO",
        "LIDERANCA",
        "OPERACAO",
        "ATIVIDADE",
        "DATA ULT. ADM",
    ]
].copy()

base = base.merge(
    ativos_ref,
    on="CHAVE_COLABORADOR",
    how="left",
    suffixes=("", "_ATIVOS")
)

# =========================
# Tratamento dos campos vindos da base ativos
# =========================
for col in ["CARGO", "LIDERANCA", "OPERACAO", "ATIVIDADE", "DATA ULT. ADM"]:
    base = garantir_coluna(base, col, "")

base["COLABORADOR"] = base["COLABORADOR"].astype(str).str.strip()

for col in ["CARGO", "LIDERANCA", "OPERACAO", "ATIVIDADE"]:
    base[col] = base[col].astype(str).str.strip()
    base.loc[base[col].isin(["", "nan", "None", "NONE"]), col] = pd.NA

base["CARGO"] = base["CARGO"].fillna("Sem cargo")
base["LIDERANCA"] = base["LIDERANCA"].fillna("Sem liderança")
base["OPERACAO"] = base["OPERACAO"].fillna("Sem operação")
base["ATIVIDADE"] = base["ATIVIDADE"].fillna("Sem atividade")

# =========================
# Datas + métricas
# =========================
base = tratar_data_adm(base, "DATA ULT. ADM")

for col in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS", "HABILIDADES EMPODERAMENTO"]:
    base = garantir_coluna(base, col, 0)
    base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)

base["PRAZO_TECNICAS_DT"] = base["DATA ULT. ADM"] + pd.Timedelta(days=30)
base["PRAZO_ESPECIFICAS_DT"] = base["DATA ULT. ADM"] + pd.Timedelta(days=60)

base["PRAZO TECNICAS"] = base["PRAZO_TECNICAS_DT"].dt.strftime("%d/%m/%Y").fillna("")
base["PRAZO ESPECIFICAS"] = base["PRAZO_ESPECIFICAS_DT"].dt.strftime("%d/%m/%Y").fillna("")

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

base["STATUS TECNICAS"] = base.apply(status_tecnicas, axis=1)
base["STATUS ESPECIFICAS"] = base.apply(status_especificas, axis=1)

# =========================
# Padronização extra
# =========================
base = garantir_coluna(base, "NIVEIS", "")
base["NIVEIS"] = base["NIVEIS"].astype(str).str.strip()
base.loc[base["NIVEIS"].isin(["", "nan", "None", "NONE"]), "NIVEIS"] = pd.NA
base["NIVEIS"] = base["NIVEIS"].fillna("Sem nível")

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
    "HABILIDADES EMPODERAMENTO",
    "NIVEIS",
    "HABILIDADE TECNICA",
    "HABILIDADE ESPECIFICA",
    "HABILIDADE EMPODERAMENTO",
    "CHAVE_COLABORADOR",
    "PRAZO_TECNICAS_DT",
    "PRAZO_ESPECIFICAS_DT",
    "DATA ULT. ADM",
]
ordem = [c for c in ordem if c in base.columns]
base = base[ordem].copy()

# =========================
# Filtros
# =========================
st.sidebar.header("Filtros")

etapas_map = {"Técnicas": "TECNICAS", "Específicas": "ESPECIFICAS", "Empoderamento": "EMPODERAMENTO"}
f_etapas = st.sidebar.multiselect(
    "Etapa (para o gráfico de %)",
    list(etapas_map.keys()),
    default=list(etapas_map.keys())
)
st.sidebar.caption("📌 Cards, Vencimento e Vencida consideram apenas Técnicas + Específicas.")

f_operacao = st.sidebar.multiselect("Operação", opcoes(base, "OPERACAO"))
f_lideranca = st.sidebar.multiselect("Liderança", opcoes(base, "LIDERANCA"))
f_atividade = st.sidebar.multiselect("Atividade", opcoes(base, "ATIVIDADE"))
f_niveis = st.sidebar.multiselect("Níveis", opcoes(base, "NIVEIS"))
f_status = st.sidebar.multiselect("Status (Téc/Espec)", ["Realizado", "Não realizado", "No prazo"])

# =========================
# Filtro de período por admissão
# sempre do mais antigo ao mais atual da base
# =========================
base["_ADM_DT"] = pd.to_datetime(base["DATA ULT. ADM"], errors="coerce", dayfirst=True)

# considera só datas válidas
adm_validas = base.loc[base["_ADM_DT"].notna(), "_ADM_DT"].copy()

if adm_validas.empty:
    min_adm = pd.Timestamp(datetime.today().date())
    max_adm = pd.Timestamp(datetime.today().date())
else:
    min_adm = adm_validas.min().normalize()
    max_adm = adm_validas.max().normalize()

st.sidebar.subheader("Período de admissão")
col_ini, col_fim = st.sidebar.columns(2)

with col_ini:
    data_ini = st.date_input(
        "Início",
        value=min_adm.date(),
        min_value=min_adm.date(),
        max_value=max_adm.date(),
        format="DD/MM/YYYY",
        key="skap_ini",
    )

with col_fim:
    data_fim = st.date_input(
        "Fim",
        value=max_adm.date(),
        min_value=min_adm.date(),
        max_value=max_adm.date(),
        format="DD/MM/YYYY",
        key="skap_fim",
    )

if data_ini > data_fim:
    st.sidebar.warning("⚠️ A data de Início não pode ser maior que a data de Fim. Ajustei automaticamente.")
    data_ini, data_fim = data_fim, data_ini

# =========================
# Aplicação dos filtros
# =========================
base_f = base.copy()

if f_operacao:
    base_f = base_f[base_f["OPERACAO"].isin(f_operacao)]
if f_lideranca:
    base_f = base_f[base_f["LIDERANCA"].isin(f_lideranca)]
if f_atividade:
    base_f = base_f[base_f["ATIVIDADE"].isin(f_atividade)]
if f_niveis:
    base_f = base_f[base_f["NIVEIS"].isin(f_niveis)]

# filtro de data por admissão
base_f["_ADM_DT"] = pd.to_datetime(base_f["DATA ULT. ADM"], errors="coerce", dayfirst=True)

data_ini_ts = pd.Timestamp(data_ini).normalize()
data_fim_ts = pd.Timestamp(data_fim).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

base_f = base_f[
    base_f["_ADM_DT"].notna() &
    (base_f["_ADM_DT"] >= data_ini_ts) &
    (base_f["_ADM_DT"] <= data_fim_ts)
]

if f_status:
    base_f = base_f[
        base_f["STATUS TECNICAS"].isin(f_status) |
        base_f["STATUS ESPECIFICAS"].isin(f_status)
    ]

# =========================
# Máscaras
# =========================
def mask_pendencia(df: pd.DataFrame) -> pd.Series:
    return (
        (df["STATUS TECNICAS"] == "Não realizado") |
        (df["STATUS ESPECIFICAS"] == "Não realizado")
    )

def mask_concluido(df: pd.DataFrame) -> pd.Series:
    return (
        (df["STATUS TECNICAS"] == "Realizado") &
        (df["STATUS ESPECIFICAS"] == "Realizado")
    )

def mask_no_prazo(df: pd.DataFrame) -> pd.Series:
    return (
        (df["STATUS TECNICAS"] == "No prazo") |
        (df["STATUS ESPECIFICAS"] == "No prazo")
    )

# =========================
# Cards
# =========================
total = len(base_f)
pend = int(mask_pendencia(base_f).sum())
concl = int(mask_concluido(base_f).sum())
nop = int(mask_no_prazo(base_f).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", total)
c2.metric("✅ 100% concluídos (Téc/Espec)", concl)
c3.metric("🟡 No prazo (Téc/Espec)", nop)
c4.metric("🔴 Com pendência (Téc/Espec)", pend)

st.divider()

# =========================
# 📈 % de realização por Unidade e por Etapa
# =========================
st.subheader("📈 % de realização por Unidade e por Etapa")

considera_tec_graf = "Técnicas" in f_etapas
considera_esp_graf = "Específicas" in f_etapas
considera_emp_graf = "Empoderamento" in f_etapas

def pct_tec(df):
    if len(df) == 0:
        return 0.0
    return (df["STATUS TECNICAS"].isin(["Realizado", "No prazo"]).sum()) / len(df)

def pct_esp(df):
    if len(df) == 0:
        return 0.0
    return (df["STATUS ESPECIFICAS"].isin(["Realizado", "No prazo"]).sum()) / len(df)

def pct_emp_realizado(df):
    if len(df) == 0:
        return 0.0
    emp = pd.to_numeric(df["HABILIDADES EMPODERAMENTO"], errors="coerce").fillna(0)
    return (emp > 0).mean()

linhas = []
for op, g in base_f.groupby("OPERACAO", dropna=False):
    g = g.drop_duplicates(subset=["COLABORADOR"], keep="first").copy()

    if considera_tec_graf:
        linhas.append({"OPERACAO": op, "ETAPA": "Técnicas", "PERCENTUAL": pct_tec(g)})
    if considera_esp_graf:
        linhas.append({"OPERACAO": op, "ETAPA": "Específicas", "PERCENTUAL": pct_esp(g)})
    if considera_emp_graf:
        linhas.append({"OPERACAO": op, "ETAPA": "Empoderamento", "PERCENTUAL": pct_emp_realizado(g)})

pct_df = pd.DataFrame(linhas)

if pct_df.empty:
    st.info("Sem dados para calcular o percentual com os filtros atuais.")
else:
    pct_df["PERCENTUAL_TXT"] = pct_df["PERCENTUAL"].map(lambda x: f"{x:.0%}")
    fig_pct = px.bar(
        pct_df.sort_values(["OPERACAO", "ETAPA"]),
        x="OPERACAO",
        y="PERCENTUAL",
        color="ETAPA",
        text="PERCENTUAL_TXT",
        barmode="group",
    )
    fig_pct.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_pct, use_container_width=True)

st.divider()

# =========================
# Base auxiliar para alertas
# =========================
hoje = pd.to_datetime(datetime.today().date())

tmp_cols = [
    "COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA",
    "HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS",
    "PRAZO_TECNICAS_DT", "PRAZO_ESPECIFICAS_DT",
    "STATUS TECNICAS", "STATUS ESPECIFICAS",
    "CHAVE_COLABORADOR"
]
tmp_cols = [c for c in tmp_cols if c in base.columns]
tmp = base[tmp_cols].copy()

tmp = tmp.merge(
    base_f[["CHAVE_COLABORADOR"]].drop_duplicates(),
    on="CHAVE_COLABORADOR",
    how="inner"
)

tmp = tmp.drop_duplicates(subset=["COLABORADOR"], keep="first")

# =========================
# Atenção: Vencimento próximo
# =========================
alertas = []

mask_tec = (tmp["STATUS TECNICAS"] != "Realizado") & tmp["PRAZO_TECNICAS_DT"].notna()
dias_tec = (tmp.loc[mask_tec, "PRAZO_TECNICAS_DT"] - hoje).dt.days
tmp_tec = tmp.loc[mask_tec].copy()
tmp_tec["DIAS PARA VENCER"] = dias_tec.values
tmp_tec = tmp_tec[(tmp_tec["DIAS PARA VENCER"] >= 0) & (tmp_tec["DIAS PARA VENCER"] <= 7)]
if len(tmp_tec) > 0:
    tmp_tec["ETAPA"] = "Técnicas"
    tmp_tec["DATA VENCIMENTO"] = tmp_tec["PRAZO_TECNICAS_DT"].dt.strftime("%d/%m/%Y")
    alertas.append(tmp_tec[["COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA", "ETAPA", "DATA VENCIMENTO", "DIAS PARA VENCER"]])

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
    alerta_df = alerta_df.sort_values(
        ["DIAS PARA VENCER", "OPERACAO", "LIDERANCA", "COLABORADOR"],
        ascending=[True, True, True, True]
    )

st.subheader("⚠️ Atenção: Vencimento próximo (Téc/Espec)")
if len(alerta_df) == 0:
    st.info("Nenhuma etapa vencendo nos próximos 7 dias com os filtros atuais.")
else:
    st.dataframe(centralizar_tabela(alerta_df), use_container_width=True)
    excel_alerta = preparar_excel_para_download(alerta_df, sheet_name="Vencimento_proximo")
    st.download_button(
        label="⬇️ Baixar Excel (Vencimento próximo)",
        data=excel_alerta,
        file_name="vencimento_proximo_skap.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.divider()

# =========================
# 🔴 Atenção: Skap Vencida
# =========================
vencidas = []

mask_vtec = (tmp["STATUS TECNICAS"] == "Não realizado") & tmp["PRAZO_TECNICAS_DT"].notna()
dias_vtec = (hoje - tmp.loc[mask_vtec, "PRAZO_TECNICAS_DT"]).dt.days
tmp_vtec = tmp.loc[mask_vtec].copy()
tmp_vtec["DIAS VENCIDO"] = dias_vtec.values
tmp_vtec = tmp_vtec[tmp_vtec["DIAS VENCIDO"] > 0]
if len(tmp_vtec) > 0:
    tmp_vtec["ETAPA"] = "Técnicas"
    tmp_vtec["DATA VENCIMENTO"] = tmp_vtec["PRAZO_TECNICAS_DT"].dt.strftime("%d/%m/%Y")
    vencidas.append(tmp_vtec[["COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA", "ETAPA", "DATA VENCIMENTO", "DIAS VENCIDO"]])

mask_vesp = (tmp["STATUS ESPECIFICAS"] == "Não realizado") & tmp["PRAZO_ESPECIFICAS_DT"].notna()
dias_vesp = (hoje - tmp.loc[mask_vesp, "PRAZO_ESPECIFICAS_DT"]).dt.days
tmp_vesp = tmp.loc[mask_vesp].copy()
tmp_vesp["DIAS VENCIDO"] = dias_vesp.values
tmp_vesp = tmp_vesp[tmp_vesp["DIAS VENCIDO"] > 0]
if len(tmp_vesp) > 0:
    tmp_vesp["ETAPA"] = "Específicas"
    tmp_vesp["DATA VENCIMENTO"] = tmp_vesp["PRAZO_ESPECIFICAS_DT"].dt.strftime("%d/%m/%Y")
    vencidas.append(tmp_vesp[["COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA", "ETAPA", "DATA VENCIMENTO", "DIAS VENCIDO"]])

vencida_df = pd.concat(vencidas, ignore_index=True) if vencidas else pd.DataFrame(
    columns=["COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA", "ETAPA", "DATA VENCIMENTO", "DIAS VENCIDO"]
)

if len(vencida_df) > 0:
    vencida_df = vencida_df.sort_values(
        ["DIAS VENCIDO", "OPERACAO", "LIDERANCA", "COLABORADOR"],
        ascending=[False, True, True, True]
    )

st.subheader("🔴 Atenção: Skap Vencida (Téc/Espec)")
if len(vencida_df) == 0:
    st.info("Nenhuma etapa vencida (Não realizado) com os filtros atuais.")
else:
    st.write(f"Total de etapas vencidas no filtro atual: **{len(vencida_df)}**")
    st.dataframe(
        centralizar_tabela(vencida_df).map(
            lambda v: "color: red; font-weight:700;",
            subset=["DIAS VENCIDO"]
        ),
        use_container_width=True
    )
    excel_vencida = preparar_excel_para_download(vencida_df, sheet_name="Skap_vencida")
    st.download_button(
        label="⬇️ Baixar Excel (Skap vencida)",
        data=excel_vencida,
        file_name="skap_vencida.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.divider()

# =========================
# 🔴 Pendências por Operação/Liderança
# =========================
st.subheader("🔴 Pendências (Não realizado) por Operação (Téc/Espec)")

vencida_graf = vencida_df.drop_duplicates(
    subset=["COLABORADOR", "ETAPA", "DATA VENCIMENTO"],
    keep="first"
).copy()

gop = (
    vencida_graf.groupby("OPERACAO", dropna=False)
    .size()
    .sort_values(ascending=False)
    .reset_index(name="Quantidade")
)

if len(gop) == 0:
    st.info("Sem etapas vencidas no filtro atual.")
else:
    fig1 = px.bar(gop, x="OPERACAO", y="Quantidade", text="Quantidade")
    fig1.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig1, use_container_width=True)

st.subheader("🔴 Pendências (Não realizado) por Liderança (Téc/Espec)")

glid = (
    vencida_graf.groupby("LIDERANCA", dropna=False)
    .size()
    .sort_values(ascending=False)
    .reset_index(name="Quantidade")
)

if len(glid) == 0:
    st.info("Sem etapas vencidas no filtro atual.")
else:
    fig2 = px.bar(glid, x="LIDERANCA", y="Quantidade", text="Quantidade")
    fig2.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =========================
# Tabela principal
# =========================
st.subheader("📋 Detalhamento Individual")

tabela_raw = base_f.drop(
    columns=["_ADM_DT", "CHAVE_COLABORADOR", "PRAZO_TECNICAS_DT", "PRAZO_ESPECIFICAS_DT", "DATA ULT. ADM"],
    errors="ignore"
).copy()

tabela = tabela_raw.copy()

if "DATA ADMISSAO" in tabela.columns:
    tabela["_DATA_ADMISSAO_DT"] = pd.to_datetime(tabela["DATA ADMISSAO"], errors="coerce", dayfirst=True)
    tabela = tabela.sort_values("_DATA_ADMISSAO_DT", ascending=True).drop(columns=["_DATA_ADMISSAO_DT"])

for c in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS", "HABILIDADES EMPODERAMENTO"]:
    if c in tabela.columns:
        tabela[c] = pd.to_numeric(tabela[c], errors="coerce").fillna(0).map(lambda x: f"{x:.0%}")

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

excel_detalhe = preparar_excel_para_download(tabela_raw, sheet_name="Detalhamento")
st.download_button(
    label="⬇️ Baixar Excel (Detalhamento individual)",
    data=excel_detalhe,
    file_name="detalhamento_individual_skap.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

st.divider()

# =========================
# 🌡️Farol da Skap > Termômetro de Gente
# =========================
st.subheader("🌡️Farol da Skap > Termômetro de Gente")

# base do farol respeitando os filtros já aplicados na página
farol_base = base_f.copy()

# garante numérico
farol_base["HABILIDADES TECNICAS"] = pd.to_numeric(
    farol_base["HABILIDADES TECNICAS"], errors="coerce"
).fillna(0)

# mais de 6 meses de casa
farol_6m = farol_base[farol_base["TEMPO DE CASA"] > 180].copy()

# aderentes = técnicas >= 100%
farol_6m["ADERENTE_TEC"] = farol_6m["HABILIDADES TECNICAS"] >= 1

# pendentes = técnicas < 100%
pendentes_6m = farol_6m[farol_6m["HABILIDADES TECNICAS"] < 1].copy()

# -------------------------
# Card
# -------------------------
card_total_pend_6m = len(pendentes_6m)

c_farol_1, c_farol_2 = st.columns([1, 3])
with c_farol_1:
    st.metric(
        "👥 +6 meses com Técnicas < 100%",
        card_total_pend_6m
    )

# -------------------------
# Gráfico de aderência por unidade
# aderência = técnicas >= 100% / total com mais de 6 meses
# -------------------------
aderencia_unidade = (
    farol_6m.groupby("OPERACAO", dropna=False)
    .agg(
        TOTAL_6M=("COLABORADOR", "count"),
        ADERENTES=("ADERENTE_TEC", "sum")
    )
    .reset_index()
)

if not aderencia_unidade.empty:
    aderencia_unidade["ADERENCIA"] = np.where(
        aderencia_unidade["TOTAL_6M"] > 0,
        aderencia_unidade["ADERENTES"] / aderencia_unidade["TOTAL_6M"],
        0
    )
    aderencia_unidade["ADERENCIA_TXT"] = aderencia_unidade["ADERENCIA"].map(lambda x: f"{x:.0%}")

    st.markdown("**📈 Aderência de Habilidades Técnicas por Unidade (+6 meses de casa)**")

    fig_farol = px.bar(
        aderencia_unidade.sort_values("ADERENCIA", ascending=False),
        x="OPERACAO",
        y="ADERENCIA",
        text="ADERENCIA_TXT",
        hover_data={
            "TOTAL_6M": True,
            "ADERENTES": True,
            "ADERENCIA": ':.1%'
        }
    )
    fig_farol.update_layout(
        yaxis_tickformat=".0%",
        xaxis_title="Unidade",
        yaxis_title="Aderência"
    )
    st.plotly_chart(fig_farol, use_container_width=True)
else:
    st.info("Não há colaboradores com mais de 6 meses de casa nos filtros atuais para calcular a aderência por unidade.")

# -------------------------
# Tabela de pendências
# -------------------------
st.markdown("**📋 Pessoas com mais de 6 meses de casa e Habilidades Técnicas < 100%**")

cols_pend_6m = [
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
]

cols_pend_6m = [c for c in cols_pend_6m if c in pendentes_6m.columns]

tabela_pend_6m_raw = pendentes_6m[cols_pend_6m].copy()

tabela_pend_6m = tabela_pend_6m_raw.copy()
if "HABILIDADES TECNICAS" in tabela_pend_6m.columns:
    tabela_pend_6m["HABILIDADES TECNICAS"] = pd.to_numeric(
        tabela_pend_6m["HABILIDADES TECNICAS"], errors="coerce"
    ).fillna(0).map(lambda x: f"{x:.0%}")

if "STATUS TECNICAS" in tabela_pend_6m.columns:
    tabela_pend_6m["STATUS TECNICAS"] = tabela_pend_6m["STATUS TECNICAS"].astype(str).map(
        lambda s: "🔴 Não realizado" if s == "Não realizado"
        else "🟢 Realizado" if s == "Realizado"
        else "🟡 No prazo" if s == "No prazo"
        else s
    )

if tabela_pend_6m.empty:
    st.success("Nenhuma pendência encontrada para colaboradores com mais de 6 meses de casa.")
else:
    st.dataframe(centralizar_tabela(tabela_pend_6m), use_container_width=True)

    excel_pend_6m = preparar_excel_para_download(
        tabela_pend_6m_raw,
        sheet_name="Farol_Termometro_Gente"
    )

    st.download_button(
        label="⬇️ Baixar Excel (Farol da Skap - Termômetro de Gente)",
        data=excel_pend_6m,
        file_name="farol_skap_termometro_gente.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    

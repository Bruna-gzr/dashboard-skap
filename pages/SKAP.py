import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
import numpy as np
from io import BytesIO
from zoneinfo import ZoneInfo

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

# =========================
# Última atualização dos dados + cache que invalida quando arquivo muda
# =========================
def get_last_mtime():
    arquivos = [ARQ_SKAP, ARQ_COM]
    return max(a.stat().st_mtime for a in arquivos if a.exists())

@st.cache_data(show_spinner=True)
def carregar_dados(_cache_key: float):
    if not ARQ_SKAP.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_SKAP}")
    if not ARQ_COM.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_COM}")
    skap_df = pd.read_excel(ARQ_SKAP)
    com_df = pd.read_excel(ARQ_COM)
    return skap_df, com_df

try:
    last_mtime = get_last_mtime()
    dt = datetime.fromtimestamp(last_mtime, tz=ZoneInfo("America/Sao_Paulo"))
    st.caption(f"🕒 Última atualização dos dados: {dt.strftime('%d/%m/%Y %H:%M')}")
    skap, comentarios = carregar_dados(last_mtime)
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
# Normalização + colunas mínimas
# =========================
skap = normalizar_colunas(skap)
comentarios = normalizar_colunas(comentarios)

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

for col in ["HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS", "HABILIDADES EMPODERAMENTO"]:
    skap = garantir_coluna(skap, col, 0)
    skap[col] = pd.to_numeric(skap[col], errors="coerce").fillna(0)

skap["PRAZO_TECNICAS_DT"] = skap["DATA ULT. ADM"] + pd.Timedelta(days=30)
skap["PRAZO_ESPECIFICAS_DT"] = skap["DATA ULT. ADM"] + pd.Timedelta(days=60)

skap["PRAZO TECNICAS"] = skap["PRAZO_TECNICAS_DT"].dt.strftime("%d/%m/%Y").fillna("")
skap["PRAZO ESPECIFICAS"] = skap["PRAZO_ESPECIFICAS_DT"].dt.strftime("%d/%m/%Y").fillna("")

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
# Padronização de campos-chave (evita divergências e NaN)
# =========================
for col in ["OPERACAO", "LIDERANCA", "ATIVIDADE"]:
    base[col] = base[col].astype(str).str.strip()
    base.loc[base[col].isin(["", "nan", "None", "NONE"]), col] = pd.NA

base["OPERACAO"] = base["OPERACAO"].fillna("Sem operação")
base["LIDERANCA"] = base["LIDERANCA"].fillna("Sem liderança")
base["ATIVIDADE"] = base["ATIVIDADE"].fillna("Sem atividade")

# =========================
# Ordem das colunas (EXATA)
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

# Filtro por etapa (apenas para gráfico de % por etapa)
etapas_map = {"Técnicas": "TECNICAS", "Específicas": "ESPECIFICAS", "Empoderamento": "EMPODERAMENTO"}
f_etapas = st.sidebar.multiselect(
    "Etapa (para o gráfico de %)",
    list(etapas_map.keys()),
    default=list(etapas_map.keys())
)
st.sidebar.caption("📌 Cards, Pendências, Vencimento e Vencida consideram apenas Técnicas + Específicas.")

f_operacao = st.sidebar.multiselect("Operação", opcoes(base, "OPERACAO"))
f_lideranca = st.sidebar.multiselect("Liderança", opcoes(base, "LIDERANCA"))
f_atividade = st.sidebar.multiselect("Atividade", opcoes(base, "ATIVIDADE"))
f_niveis = st.sidebar.multiselect("Níveis", opcoes(base, "NIVEIS"))
f_status = st.sidebar.multiselect("Status (Téc/Espec)", ["Realizado", "Não realizado", "No prazo"])

# Filtro de data por mês/ano baseado em DATA ADMISSAO
meses_pt = {
    1: "01 - Jan", 2: "02 - Fev", 3: "03 - Mar", 4: "04 - Abr",
    5: "05 - Mai", 6: "06 - Jun", 7: "07 - Jul", 8: "08 - Ago",
    9: "09 - Set", 10: "10 - Out", 11: "11 - Nov", 12: "12 - Dez"
}

base["_ADM_DT"] = pd.to_datetime(base["DATA ADMISSAO"], errors="coerce", dayfirst=True)
base["_ADM_ANO"] = base["_ADM_DT"].dt.year
base["_ADM_MES"] = base["_ADM_DT"].dt.month

anos_disp = sorted([int(x) for x in base["_ADM_ANO"].dropna().unique().tolist()])
anos_opts = ["Todos"] + anos_disp
mes_opts = ["Todos"] + [meses_pt[m] for m in range(1, 13)]

f_ano = st.sidebar.selectbox("Ano de admissão", anos_opts, index=0)
f_mes_lbl = st.sidebar.selectbox("Mês de admissão", mes_opts, index=0)

# -------------------------
# Aplicação dos filtros
# -------------------------
base_f = base.copy()

if f_operacao:
    base_f = base_f[base_f["OPERACAO"].isin(f_operacao)]
if f_lideranca:
    base_f = base_f[base_f["LIDERANCA"].isin(f_lideranca)]
if f_atividade:
    base_f = base_f[base_f["ATIVIDADE"].isin(f_atividade)]
if f_niveis:
    base_f = base_f[base_f["NIVEIS"].isin(f_niveis)]

# Filtro de ano/mês (DATA ADMISSAO)
if f_ano != "Todos":
    base_f = base_f[base_f["_ADM_ANO"] == int(f_ano)]

if f_mes_lbl != "Todos":
    mes_num = int(f_mes_lbl.split(" - ")[0])
    base_f = base_f[base_f["_ADM_MES"] == mes_num]

# Filtro de status (somente Técnicas/Específicas)
if f_status:
    base_f = base_f[
        base_f["STATUS TECNICAS"].isin(f_status) |
        base_f["STATUS ESPECIFICAS"].isin(f_status)
    ]

# =========================
# Máscaras (somente Técnicas + Específicas)
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
# % de realização por Unidade e por Etapa
# - Téc/Espec: (Realizado + No prazo) / total
# - Emp: média do % realizado
# =========================
st.subheader("📈 % de realização por Unidade e por Etapa")

considera_tec_graf = "Técnicas" in f_etapas
considera_esp_graf = "Específicas" in f_etapas
considera_emp_graf = "Empoderamento" in f_etapas

def pct_tec(df):
    if len(df) == 0:
        return 0.0
    return ((df["STATUS TECNICAS"].isin(["Realizado", "No prazo"])).sum()) / len(df)

def pct_esp(df):
    if len(df) == 0:
        return 0.0
    return ((df["STATUS ESPECIFICAS"].isin(["Realizado", "No prazo"])).sum()) / len(df)

def pct_emp(df):
    if len(df) == 0:
        return 0.0
    return pd.to_numeric(df["HABILIDADES EMPODERAMENTO"], errors="coerce").fillna(0).mean()

linhas = []
for op, g in base_f.groupby("OPERACAO", dropna=False):
    if considera_tec_graf:
        linhas.append({"OPERACAO": op, "ETAPA": "Técnicas", "PERCENTUAL": pct_tec(g)})
    if considera_esp_graf:
        linhas.append({"OPERACAO": op, "ETAPA": "Específicas", "PERCENTUAL": pct_esp(g)})
    if considera_emp_graf:
        linhas.append({"OPERACAO": op, "ETAPA": "Empoderamento", "PERCENTUAL": pct_emp(g)})

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
# Gráficos de Pendência (somente Téc/Espec)
# =========================
st.subheader("🔴 Pendências (Não realizado) por Operação (Téc/Espec)")
pend_df = base_f[mask_pendencia(base_f)].copy()

gop = (
    pend_df.groupby("OPERACAO", dropna=False)
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

st.subheader("🔴 Pendências (Não realizado) por Liderança (Téc/Espec)")
glid = (
    pend_df.groupby("LIDERANCA", dropna=False)
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
# Atenção: Vencimento próximo (até 7 dias) - Somente Téc/Espec
# =========================
st.divider()

hoje = pd.to_datetime(datetime.today().date())

tmp = skap[[
    "COLABORADOR", "CARGO", "OPERACAO", "LIDERANCA",
    "HABILIDADES TECNICAS", "HABILIDADES ESPECIFICAS",
    "PRAZO_TECNICAS_DT", "PRAZO_ESPECIFICAS_DT",
    "STATUS TECNICAS", "STATUS ESPECIFICAS"
]].copy()

tmp = normalizar_colunas(tmp)
tmp = tmp.merge(
    base_f[["COLABORADOR"]].drop_duplicates(),
    on="COLABORADOR",
    how="inner"
)

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
    alerta_df = alerta_df.sort_values(["DIAS PARA VENCER", "OPERACAO", "LIDERANCA", "COLABORADOR"],
                                      ascending=[True, True, True, True])

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
# 🔴 Atenção: Skap Vencida (Téc/Espec)
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
    vencida_df = vencida_df.sort_values(["DIAS VENCIDO", "OPERACAO", "LIDERANCA", "COLABORADOR"],
                                        ascending=[False, True, True, True])

st.subheader("🔴 Atenção: Skap Vencida (Téc/Espec)")
if len(vencida_df) == 0:
    st.info("Nenhuma etapa vencida (Não realizado) com os filtros atuais.")
else:
    st.write(f"Total de etapas vencidas no filtro atual: **{len(vencida_df)}**")
    st.dataframe(
        centralizar_tabela(vencida_df).applymap(
            lambda v: "color: red; font-weight:700;", subset=["DIAS VENCIDO"]
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
# Tabela principal
# =========================
st.subheader("📋 Detalhamento Individual")

# 1) Base RAW (para exportar)
tabela_raw = base_f.drop(columns=["_ADM_DT", "_ADM_ANO", "_ADM_MES"], errors="ignore").copy()

# 2) Base DISPLAY (para mostrar na tela)
tabela = tabela_raw.copy()

# Ordenação por admissão (na tela)
if "DATA ADMISSAO" in tabela.columns:
    tabela["_DATA_ADMISSAO_DT"] = pd.to_datetime(tabela["DATA ADMISSAO"], errors="coerce", dayfirst=True)
    tabela = tabela.sort_values("_DATA_ADMISSAO_DT", ascending=True).drop(columns=["_DATA_ADMISSAO_DT"])

# Formatar % SOMENTE no display (tela)
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

# Tela
st.dataframe(centralizar_tabela(tabela), use_container_width=True)

# Excel: exporta o RAW (sem % em string)
excel_detalhe = preparar_excel_para_download(tabela_raw, sheet_name="Detalhamento")
st.download_button(
    label="⬇️ Baixar Excel (Detalhamento individual)",
    data=excel_detalhe,
    file_name="detalhamento_individual_skap.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

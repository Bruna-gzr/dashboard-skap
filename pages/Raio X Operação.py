import streamlit as st

st.set_page_config(
    page_title="RAIO X OPERAÇÃO",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path
from io import BytesIO
from zoneinfo import ZoneInfo
import unicodedata
import re

# =========================================================
# TÍTULO (1 linha)
# =========================================================
st.markdown(
    "<h2 style='text-align:center; margin:0;'>RAIO X OPERAÇÃO</h2>",
    unsafe_allow_html=True
)
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# =========================================================
# ARQUIVOS (pasta data/)
# =========================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"              # BASE PRINCIPAL
ARQ_SKAP = DATA_DIR / "Skap.xlsx"                                     # pegar coluna NIVEIS
ARQ_PRONT = DATA_DIR / "Prontuario Condutor.xlsx"                     # Nome + Pontuação Ponderada
ARQ_VALES = DATA_DIR / "Vales.xlsx"                                   # COLABORADOR + DATA
ARQ_RV = DATA_DIR / "RV.xlsx"                                         # NOME + DATA + TOTAL RV + % + RECARGA
ARQ_ABS = DATA_DIR / "Absenteismo.xlsx"                               # COLABORADOR + DATA + Tipo de Ausência
ARQ_ACID = DATA_DIR / "Ocorrencia de acidentes.xlsx"                  # COLABORADOR + Data da Ocorrência
ARQ_DTO = DATA_DIR / "Desvios de DTOs.xlsx"                           # COLABORADOR + Data
ARQ_IND = DATA_DIR / "Resultados indicadores operacionais.xlsx"       # COLABORADOR + Data + PDV + BEES + TML (MIN) + JL + STATUS

# =========================================================
# ATUALIZAÇÃO + CACHE BUSTER + BOTÃO REFRESH
# =========================================================
def get_last_mtime():
    arquivos = [ARQ_ATIVOS, ARQ_SKAP, ARQ_PRONT, ARQ_VALES, ARQ_RV, ARQ_ABS, ARQ_ACID, ARQ_DTO, ARQ_IND]
    mtimes = [a.stat().st_mtime for a in arquivos if a.exists()]
    return max(mtimes) if mtimes else None

last_mtime = get_last_mtime()

try:
    if last_mtime is not None:
        dt = datetime.fromtimestamp(last_mtime, tz=ZoneInfo("America/Sao_Paulo"))
        st.caption(f"🕒 Última atualização dos dados: {dt.strftime('%d/%m/%Y %H:%M')}")
    else:
        st.caption("🕒 Última atualização: não disponível")
except Exception:
    st.caption("🕒 Última atualização: não disponível")

c_refresh, _ = st.columns([1, 5])
with c_refresh:
    if st.button("🔄 Atualizar dados agora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =========================================================
# UTILS
# =========================================================
def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        unicodedata.normalize("NFKD", str(c).strip()).encode("ascii", "ignore").decode("utf-8").upper()
        for c in df.columns
    ]
    return df

def normalizar_nome(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = s.upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tratar_data_segura(series: pd.Series) -> pd.Series:
    dt_txt = pd.to_datetime(series, errors="coerce", dayfirst=True)

    num = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = num.notna() & np.isfinite(num) & (num >= 20000) & (num <= 80000)

    dt_excel = pd.Series(pd.NaT, index=series.index)
    if mask.any():
        dt_excel.loc[mask] = pd.to_datetime(
            num.loc[mask].astype("int64"),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return dt_txt.fillna(dt_excel)

def to_month_key(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    return d.dt.to_period("M").astype(str)

def month_label(pt_period: str) -> str:
    try:
        y, m = pt_period.split("-")
        return f"{m}/{y}"
    except Exception:
        return pt_period

def preparar_excel_para_download(df: pd.DataFrame, sheet_name="Dados") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()

def safe_float(x):
    return pd.to_numeric(x, errors="coerce")

def parse_percent(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("%", "").replace(",", ".")
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return np.nan
    if 0 <= v <= 1.5:
        v = v * 100
    return float(v)

def parse_time_mmss(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("0 days ", "").replace("0 day ", "")

    v = pd.to_numeric(s.replace(",", "."), errors="coerce")
    if not pd.isna(v) and re.fullmatch(r"[-+]?\d+(\.\d+)?", s.replace(",", ".")) is not None:
        return float(v)

    parts = s.split(":")
    parts = [p.strip() for p in parts if p.strip() != ""]
    if len(parts) == 2:
        mm, ss = parts
        mm = pd.to_numeric(mm, errors="coerce")
        ss = pd.to_numeric(ss, errors="coerce")
        if pd.isna(mm) or pd.isna(ss):
            return np.nan
        return float(mm) + float(ss) / 60.0
    if len(parts) == 3:
        hh, mm, ss = parts
        hh = pd.to_numeric(hh, errors="coerce")
        mm = pd.to_numeric(mm, errors="coerce")
        ss = pd.to_numeric(ss, errors="coerce")
        if pd.isna(hh) or pd.isna(mm) or pd.isna(ss):
            return np.nan
        return float(hh) * 60.0 + float(mm) + float(ss) / 60.0
    return np.nan

def fmt_pct(v):
    return "" if pd.isna(v) else f"{v:.2f}%"

def fmt_min(v):
    if pd.isna(v):
        return ""
    mm = int(v)
    ss = int(round((v - mm) * 60))
    if ss == 60:
        mm += 1
        ss = 0
    return f"{mm:02d}:{ss:02d}"

def norm_operacao(op: str) -> str:
    return normalizar_nome(op)

def centralizar_tabela(df: pd.DataFrame):
    return (
        df.style
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )

# =========================================================
# FUNÇÕES (unificação)
# =========================================================
FUNCOES_MAP = {
    normalizar_nome("Ajudante Distribuição"): "Ajudante de Distribuição",
    normalizar_nome("Ajudante AS"): "Ajudante de Distribuição",

    normalizar_nome("Motorista Caminhão Distribuição"): "Motorista de Distribuição",
    normalizar_nome("Motorista de Distribuição AS"): "Motorista de Distribuição",

    normalizar_nome("Motorista Entregador I"): "Motorista de Van",
    normalizar_nome("Motorista Entregador"): "Motorista de Van",
    normalizar_nome("Motorista Entregador II"): "Motorista de Van",

    normalizar_nome("Ajudante Armazém"): "Ajudante de armazem",
    normalizar_nome("Amarrador"): "Ajudante de armazem",

    normalizar_nome("Operador de Empilhadeira"): "Operador",
    normalizar_nome("Operador Conferente"): "Operador",
}

FUNCOES_PERMITIDAS = sorted(list(set(FUNCOES_MAP.values())))

def unificar_funcao(cargo: str) -> str:
    k = normalizar_nome(cargo)
    return FUNCOES_MAP.get(k, "")

# =========================================================
# METAS/PONTOS POR OPERAÇÃO
# =========================================================
METAS = {
    "CD CASCAVEL": {
        "PDV": {"meta": 3.5, "pontos": 5, "tipo": "lte_pct"},
        "BEES": {"meta": 95.0, "pontos": 1, "tipo": "gte_pct"},  # ✅ ajuste
        "TML": {"meta": 0.5, "pontos": 2, "tipo": "lte_min"},
        "JL": {"meta": 80.0, "pontos": 3, "tipo": "gte_pct"},
        "ABS": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "VALES": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "ACIDENTE": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "DTO": {"meta": 0, "pontos": 4, "tipo": "eq0"},
    },
    "CD DIADEMA": {
        "PDV": {"meta": 3.5, "pontos": 4, "tipo": "lte_pct"},
        "BEES": {"meta": 95.0, "pontos": 3, "tipo": "gte_pct"},
        "TML": {"meta": 0.5, "pontos": 3, "tipo": "lte_min"},
        "JL": {"meta": 80.0, "pontos": 1, "tipo": "gte_pct"},
        "ABS": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "VALES": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "ACIDENTE": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "DTO": {"meta": 0, "pontos": 4, "tipo": "eq0"},
    },
    "CD FOZ DO IGUACU": {
        "PDV": {"meta": 3.01, "pontos": 3, "tipo": "lte_pct"},
        "BEES": {"meta": 95.0, "pontos": 2, "tipo": "gte_pct"},
        "TML": {"meta": 0.5, "pontos": 3, "tipo": "lte_min"},
        "JL": {"meta": 80.0, "pontos": 3, "tipo": "gte_pct"},
        "ABS": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "VALES": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "ACIDENTE": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "DTO": {"meta": 0, "pontos": 4, "tipo": "eq0"},
    },
    "CD LONDRINA": {
        "PDV": {"meta": 3.5, "pontos": 3, "tipo": "lte_pct"},
        # BEES não aplicável
        "TML": {"meta": 0.5, "pontos": 5, "tipo": "lte_min"},
        "JL": {"meta": 80.0, "pontos": 3, "tipo": "gte_pct"},
        "ABS": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "VALES": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "ACIDENTE": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "DTO": {"meta": 0, "pontos": 4, "tipo": "eq0"},
    },
    "CD LITORAL": {
        "PDV": {"meta": 3.5, "pontos": 3, "tipo": "lte_pct"},
        "BEES": {"meta": 95.0, "pontos": 2, "tipo": "gte_pct"},
        "TML": {"meta": 0.5, "pontos": 3, "tipo": "lte_min"},
        "JL": {"meta": 80.0, "pontos": 3, "tipo": "gte_pct"},
        "ABS": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "VALES": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "ACIDENTE": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "DTO": {"meta": 0, "pontos": 4, "tipo": "eq0"},
    },
    "CD SAO CRISTOVAO": {
        "PDV": {"meta": 3.5, "pontos": 3, "tipo": "lte_pct"},
        "BEES": {"meta": 95.0, "pontos": 4, "tipo": "gte_pct"},
        "TML": {"meta": 0.5, "pontos": 3, "tipo": "lte_min"},
        "JL": {"meta": 80.0, "pontos": 1, "tipo": "gte_pct"},
        "ABS": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "VALES": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "ACIDENTE": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "DTO": {"meta": 0, "pontos": 4, "tipo": "eq0"},
    },
    "CD FRANCISCO BELTRAO": {
        "PDV": {"meta": 2.84, "pontos": 2, "tipo": "lte_pct"},
        "BEES": {"meta": 95.0, "pontos": 3, "tipo": "gte_pct"},
        "TML": {"meta": 0.5, "pontos": 3, "tipo": "lte_min"},
        "JL": {"meta": 89.0, "pontos": 3, "tipo": "gte_pct"},
        "ABS": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "VALES": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "ACIDENTE": {"meta": 0, "pontos": 5, "tipo": "eq0"},
        "DTO": {"meta": 0, "pontos": 4, "tipo": "eq0"},
    },
}

# ✅ normaliza chaves do METAS
METAS = {norm_operacao(k): v for k, v in METAS.items()}

# =========================================================
# PRONTUÁRIO: LER "Pontuação <br> Ponderada" (robusto)
# =========================================================
def ler_prontuario_ponderada(path_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(path_xlsx, sheet_name=0)
    df = normalizar_colunas(df)

    # procura coluna de nome
    col_nome = None
    for cand in ["EMPREGADO", "COLABORADOR", "NOME", "MOTORISTA", "CONDUTOR"]:
        if cand in df.columns:
            col_nome = cand
            break
    if col_nome is None:
        for c in df.columns:
            cc = normalizar_nome(c)
            if "NOME" in cc or "EMPREG" in cc or "COLAB" in cc:
                col_nome = c
                break
    if col_nome is None:
        return pd.DataFrame(columns=["NOME_KEY", "PRONT_PONDERADA"])

    df["NOME_KEY"] = df[col_nome].astype(str).map(normalizar_nome)

    # acha coluna ponderada (com variações)
    col_score = None
    for c in df.columns:
        cc = normalizar_nome(c)
        if "PONTUACAO" in cc and "PONDERADA" in cc:
            col_score = c
            break
        if "PONTUAC" in cc and "PONDERADA" in cc:
            col_score = c
            break
        if "PONDERADA" in cc and ("PONT" in cc or "PONTU" in cc):
            col_score = c
            break

    if col_score is None:
        return pd.DataFrame(columns=["NOME_KEY", "PRONT_PONDERADA"])

    df["PRONT_PONDERADA"] = pd.to_numeric(df[col_score], errors="coerce")
    out = (
        df[["NOME_KEY", "PRONT_PONDERADA"]]
        .dropna(subset=["NOME_KEY"])
        .drop_duplicates("NOME_KEY", keep="last")
    )
    return out

# =========================================================
# LOAD (CACHE)
# =========================================================
@st.cache_data(show_spinner=True)
def carregar_bases(_cache_key: float | None):
    if not ARQ_ATIVOS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_ATIVOS.name}")

    ativos = pd.read_excel(ARQ_ATIVOS)
    skap = pd.read_excel(ARQ_SKAP) if ARQ_SKAP.exists() else pd.DataFrame()
    vales = pd.read_excel(ARQ_VALES) if ARQ_VALES.exists() else pd.DataFrame()
    rv = pd.read_excel(ARQ_RV) if ARQ_RV.exists() else pd.DataFrame()
    abs_ = pd.read_excel(ARQ_ABS) if ARQ_ABS.exists() else pd.DataFrame()
    acid = pd.read_excel(ARQ_ACID) if ARQ_ACID.exists() else pd.DataFrame()
    dto = pd.read_excel(ARQ_DTO) if ARQ_DTO.exists() else pd.DataFrame()
    ind = pd.read_excel(ARQ_IND) if ARQ_IND.exists() else pd.DataFrame()

    pront = pd.DataFrame()
    if ARQ_PRONT.exists():
        try:
            pront = ler_prontuario_ponderada(ARQ_PRONT)
        except Exception:
            pront = pd.DataFrame()

    return ativos, skap, pront, vales, rv, abs_, acid, dto, ind

try:
    ativos, skap, pront, vales, rv, abs_, acid, dto, ind = carregar_bases(last_mtime)
except Exception as e:
    st.error(f"❌ Erro ao carregar bases: {e}")
    st.stop()

# =========================================================
# NORMALIZAÇÃO PRINCIPAL (ATIVOS)
# =========================================================
ativos = normalizar_colunas(ativos)

for c in ["COLABORADOR", "CARGO", "OPERACAO", "ATIVIDADE", "DATA ULT. ADM"]:
    if c not in ativos.columns:
        ativos[c] = ""

if "OPERAÇÃO" in ativos.columns and "OPERACAO" not in ativos.columns:
    ativos["OPERACAO"] = ativos["OPERAÇÃO"]

ativos["NOME_KEY"] = ativos["COLABORADOR"].astype(str).map(normalizar_nome)
ativos["OPER_KEY"] = ativos["OPERACAO"].astype(str).map(norm_operacao)
ativos["FUNCAO"] = ativos["CARGO"].astype(str).map(unificar_funcao)
ativos["DATA_ADM_DT"] = tratar_data_segura(ativos["DATA ULT. ADM"])
ativos["ADMISSAO"] = pd.to_datetime(ativos["DATA_ADM_DT"], errors="coerce").dt.strftime("%d/%m/%Y").fillna("")

ativos = ativos[ativos["FUNCAO"].isin(FUNCOES_PERMITIDAS)].copy()

# =========================================================
# SKAP: puxar NÍVEIS
# =========================================================
skap = normalizar_colunas(skap) if not skap.empty else skap
if not skap.empty:
    if "COLABORADOR" not in skap.columns:
        skap["COLABORADOR"] = ""
    if "NIVEIS" not in skap.columns:
        skap["NIVEIS"] = ""
    skap["NOME_KEY"] = skap["COLABORADOR"].astype(str).map(normalizar_nome)
    skap_niveis = skap[["NOME_KEY", "NIVEIS"]].drop_duplicates(subset=["NOME_KEY"], keep="last")
else:
    skap_niveis = pd.DataFrame(columns=["NOME_KEY", "NIVEIS"])

# =========================================================
# PRONTUÁRIO: DF (NOME_KEY, PRONT_PONDERADA)
# =========================================================
if pront is None or pront.empty:
    pront = pd.DataFrame(columns=["NOME_KEY", "PRONT_PONDERADA"])
else:
    pront = normalizar_colunas(pront)
    if "NOME_KEY" not in pront.columns:
        pront["NOME_KEY"] = ""
    if "PRONT_PONDERADA" not in pront.columns:
        pront["PRONT_PONDERADA"] = np.nan

# =========================================================
# BASE MASTER — filtros e cards
# =========================================================
base_master = (
    ativos.merge(skap_niveis, on="NOME_KEY", how="left")
          .merge(pront[["NOME_KEY", "PRONT_PONDERADA"]], on="NOME_KEY", how="left")
)
if "NIVEIS" not in base_master.columns:
    base_master["NIVEIS"] = ""

# ✅ Prontuário só para motoristas
MOTORISTAS_OK = {
    normalizar_nome("Motorista de Distribuição"),
    normalizar_nome("Motorista de Van"),
}
base_master["FUNCAO_KEY"] = base_master["FUNCAO"].astype(str).map(normalizar_nome)
mask_motorista = base_master["FUNCAO_KEY"].isin(MOTORISTAS_OK)
base_master.loc[~mask_motorista, "PRONT_PONDERADA"] = np.nan

def faixa_pront(v: float) -> tuple[str, str]:
    if pd.isna(v):
        return ("-", "#94a3b8")
    v = float(v)
    if 0 <= v <= 10.009:
        return ("Faixa Verde", "#22c55e")
    if 10.010 <= v <= 20.009:
        return ("Faixa Amarela", "#facc15")
    if 20.010 <= v <= 30.009:
        return ("Faixa Laranja", "#fb923c")
    if 30.010 <= v <= 40.009:
        return ("Faixa Vermelha", "#ef4444")
    if v >= 40.010:
        return ("Faixa Roxa", "#a855f7")
    return ("-", "#94a3b8")

tmp = base_master["PRONT_PONDERADA"].apply(lambda x: faixa_pront(x))
base_master["PRONT_FAIXA"] = tmp.apply(lambda t: t[0])
base_master["PRONT_COR"] = tmp.apply(lambda t: t[1])

# =========================================================
# PREPARAR EVENTOS / OCORRÊNCIAS POR MÊS
# =========================================================
def prep_evento(df: pd.DataFrame, col_nome: str, col_data: str, nome_out="NOME_KEY", data_out="DT"):
    if df is None or df.empty:
        return pd.DataFrame(columns=[nome_out, data_out, "MES"])
    df = normalizar_colunas(df)
    if col_nome.upper() not in df.columns:
        df[col_nome.upper()] = ""
    if col_data.upper() not in df.columns:
        df[col_data.upper()] = ""

    out = df[[col_nome.upper(), col_data.upper()]].copy()
    out[nome_out] = out[col_nome.upper()].astype(str).map(normalizar_nome)
    out[data_out] = tratar_data_segura(out[col_data.upper()])
    out = out.dropna(subset=[data_out])
    out["MES"] = to_month_key(out[data_out])
    return out[[nome_out, data_out, "MES"]]

vales_ev = prep_evento(vales, "COLABORADOR", "DATA")
acid_ev = prep_evento(acid, "COLABORADOR", "DATA DA OCORRENCIA")
if acid_ev.empty:
    acid_ev = prep_evento(acid, "COLABORADOR", "DATA")
dto_ev = prep_evento(dto, "COLABORADOR", "DATA")

# Absenteísmo
abs_ev = pd.DataFrame(columns=["NOME_KEY", "DT", "MES", "TIPO"])
if abs_ is not None and not abs_.empty:
    abs_ = normalizar_colunas(abs_)
    for c in ["COLABORADOR", "DATA", "TIPO DE AUSENCIA"]:
        if c not in abs_.columns:
            abs_[c] = ""
    abs_["NOME_KEY"] = abs_["COLABORADOR"].astype(str).map(normalizar_nome)
    abs_["DT"] = tratar_data_segura(abs_["DATA"])
    abs_["TIPO"] = abs_["TIPO DE AUSENCIA"].astype(str).map(normalizar_nome)
    abs_ = abs_.dropna(subset=["DT"])
    abs_["MES"] = to_month_key(abs_["DT"])
    abs_ = abs_[abs_["TIPO"].isin([normalizar_nome("JUSTIFICADA"), normalizar_nome("NÃO JUSTIFICADA"), normalizar_nome("NAO JUSTIFICADA")])]
    abs_ev = abs_[["NOME_KEY", "DT", "MES", "TIPO"]].copy()

# RV (por mês)
rv_m = pd.DataFrame(columns=["NOME_KEY", "MES", "TOTAL_RV", "PCT_RV", "RECARGA"])
if rv is not None and not rv.empty:
    rv = normalizar_colunas(rv)
    if "NOME" not in rv.columns:
        rv["NOME"] = ""
    if "DATA" not in rv.columns:
        rv["DATA"] = ""

    col_total = "TOTAL RV" if "TOTAL RV" in rv.columns else ("TOTAL_RV" if "TOTAL_RV" in rv.columns else "")
    col_pct = "%" if "%" in rv.columns else ("PCT" if "PCT" in rv.columns else ("PERCENTUAL" if "PERCENTUAL" in rv.columns else ""))
    col_rec = "RECARGA" if "RECARGA" in rv.columns else ""

    rv["NOME_KEY"] = rv["NOME"].astype(str).map(normalizar_nome)
    rv["DT"] = tratar_data_segura(rv["DATA"])
    rv = rv.dropna(subset=["DT"])
    rv["MES"] = to_month_key(rv["DT"])

    rv["TOTAL_RV_NUM"] = safe_float(rv[col_total]) if col_total else np.nan
    rv["PCT_RV_NUM"] = rv[col_pct].apply(parse_percent) if col_pct else np.nan
    rv["RECARGA_NUM"] = safe_float(rv[col_rec]) if col_rec else np.nan

    rv_m = (
        rv.groupby(["NOME_KEY", "MES"], dropna=False)
        .agg(
            TOTAL_RV=("TOTAL_RV_NUM", "sum"),
            PCT_RV=("PCT_RV_NUM", "mean"),
            RECARGA=("RECARGA_NUM", "sum"),
        )
        .reset_index()
    )

# Indicadores operacionais (por mês) + STATUS
ind_m = pd.DataFrame(columns=["NOME_KEY", "MES", "PDV", "BEES", "TML_MIN", "JL", "STATUS"])
if ind is not None and not ind.empty:
    ind = normalizar_colunas(ind)
    for c in ["COLABORADOR", "DATA", "PDV", "BEES", "TML (MIN)", "JL", "STATUS"]:
        if c not in ind.columns:
            ind[c] = ""

    ind["NOME_KEY"] = ind["COLABORADOR"].astype(str).map(normalizar_nome)
    ind["DT"] = tratar_data_segura(ind["DATA"])
    ind = ind.dropna(subset=["DT"])
    ind["MES"] = to_month_key(ind["DT"])

    ind["PDV_NUM"] = ind["PDV"].apply(parse_percent)
    ind["BEES_NUM"] = ind["BEES"].apply(parse_percent)
    ind["TML_MIN"] = ind["TML (MIN)"].apply(parse_time_mmss)
    ind["JL_NUM"] = ind["JL"].apply(parse_percent)

    ind["STATUS_NORM"] = ind["STATUS"].astype(str).map(normalizar_nome)

    def status_final(s):
        s = normalizar_nome(s)
        if s == normalizar_nome("FERIAS"):
            return "FERIAS"
        if s in [normalizar_nome("AFASTADO"), normalizar_nome("AFS")]:
            return "AFASTADO"
        return ""

    def pick_status(series: pd.Series) -> str:
        vals = [status_final(x) for x in series.dropna().astype(str).tolist()]
        if "FERIAS" in vals:
            return "FERIAS"
        if "AFASTADO" in vals:
            return "AFASTADO"
        return ""

    ind_m = (
        ind.groupby(["NOME_KEY", "MES"], dropna=False)
        .agg(
            PDV=("PDV_NUM", "mean"),
            BEES=("BEES_NUM", "mean"),
            TML_MIN=("TML_MIN", "mean"),
            JL=("JL_NUM", "mean"),
            STATUS=("STATUS_NORM", pick_status),
        )
        .reset_index()
    )

# Contagens por mês
vales_m = vales_ev.groupby(["NOME_KEY", "MES"]).size().reset_index(name="VALES")
acid_m = acid_ev.groupby(["NOME_KEY", "MES"]).size().reset_index(name="ACIDENTE")
dto_m = dto_ev.groupby(["NOME_KEY", "MES"]).size().reset_index(name="DTO")
abs_m = abs_ev.groupby(["NOME_KEY", "MES"]).size().reset_index(name="ABS")

# =========================================================
# TABELA BASE POR MÊS (colaborador x mês)
# =========================================================
meses_sets = []
for df_ in [vales_m, acid_m, dto_m, abs_m, rv_m, ind_m]:
    if df_ is not None and not df_.empty and "MES" in df_.columns:
        meses_sets.append(df_["MES"].dropna().unique().tolist())
meses_all = sorted(list(set([m for sub in meses_sets for m in sub])))

if not meses_all:
    meses_all = [pd.Timestamp.today().to_period("M").astype(str)]

colab_univ = base_master[[
    "NOME_KEY", "COLABORADOR", "OPERACAO", "OPER_KEY", "ATIVIDADE",
    "FUNCAO", "DATA_ADM_DT", "ADMISSAO", "NIVEIS",
    "PRONT_PONDERADA", "PRONT_FAIXA", "PRONT_COR"
]].copy()
colab_univ["OPER_KEY"] = colab_univ["OPER_KEY"].fillna(colab_univ["OPERACAO"].astype(str).map(norm_operacao))

grid = colab_univ.assign(key=1).merge(pd.DataFrame({"MES": meses_all, "key": 1}), on="key").drop(columns=["key"])

def left_join(grid_, df_, cols_keep):
    if df_ is None or df_.empty:
        for c in cols_keep:
            if c not in grid_.columns:
                grid_[c] = np.nan
        return grid_
    return grid_.merge(df_[["NOME_KEY", "MES"] + cols_keep], on=["NOME_KEY", "MES"], how="left")

grid = left_join(grid, ind_m, ["PDV", "BEES", "TML_MIN", "JL", "STATUS"])
grid = left_join(grid, abs_m, ["ABS"])
grid = left_join(grid, vales_m, ["VALES"])
grid = left_join(grid, acid_m, ["ACIDENTE"])
grid = left_join(grid, dto_m, ["DTO"])
grid = left_join(grid, rv_m, ["TOTAL_RV", "PCT_RV", "RECARGA"])

for c in ["ABS", "VALES", "ACIDENTE", "DTO"]:
    if c in grid.columns:
        grid[c] = pd.to_numeric(grid[c], errors="coerce").fillna(0).astype(int)

# ✅ FILTRAR MESES ANTERIORES À ADMISSÃO
grid["_MES_DT"] = pd.to_datetime(grid["MES"] + "-01", errors="coerce")
grid["_ADM_MES_DT"] = pd.to_datetime(grid["DATA_ADM_DT"], errors="coerce").dt.to_period("M").dt.to_timestamp()
grid = grid[grid["_MES_DT"] >= grid["_ADM_MES_DT"]].copy()
grid = grid.drop(columns=["_MES_DT", "_ADM_MES_DT"])

# =========================================================
# PONTUAÇÃO
# =========================================================
def calc_pontos_row(row) -> dict:
    op_key = norm_operacao(row.get("OPERACAO", ""))
    metas = METAS.get(op_key, None)

    res = {
        "PTS_PDV": 0, "PTS_BEES": 0, "PTS_TML": 0, "PTS_JL": 0,
        "PTS_ABS": 0, "PTS_VALES": 0, "PTS_ACIDENTE": 0, "PTS_DTO": 0,
        "TOTAL_PTS": 0
    }
    if metas is None:
        return res

    if "PDV" in metas:
        v = row.get("PDV", np.nan)
        if pd.notna(v) and v <= metas["PDV"]["meta"]:
            res["PTS_PDV"] = metas["PDV"]["pontos"]

    if "BEES" in metas:
        v = row.get("BEES", np.nan)
        if pd.notna(v) and v >= metas["BEES"]["meta"]:
            res["PTS_BEES"] = metas["BEES"]["pontos"]

    if "TML" in metas:
        v = row.get("TML_MIN", np.nan)
        if pd.notna(v) and v <= metas["TML"]["meta"]:
            res["PTS_TML"] = metas["TML"]["pontos"]

    if "JL" in metas:
        v = row.get("JL", np.nan)
        if pd.notna(v) and v >= metas["JL"]["meta"]:
            res["PTS_JL"] = metas["JL"]["pontos"]

    if "ABS" in metas and int(row.get("ABS", 0)) == 0:
        res["PTS_ABS"] = metas["ABS"]["pontos"]
    if "VALES" in metas and int(row.get("VALES", 0)) == 0:
        res["PTS_VALES"] = metas["VALES"]["pontos"]
    if "ACIDENTE" in metas and int(row.get("ACIDENTE", 0)) == 0:
        res["PTS_ACIDENTE"] = metas["ACIDENTE"]["pontos"]
    if "DTO" in metas and int(row.get("DTO", 0)) == 0:
        res["PTS_DTO"] = metas["DTO"]["pontos"]

    res["TOTAL_PTS"] = sum([
        res["PTS_PDV"], res["PTS_BEES"], res["PTS_TML"], res["PTS_JL"],
        res["PTS_ABS"], res["PTS_VALES"], res["PTS_ACIDENTE"], res["PTS_DTO"]
    ])
    return res

pts_df = grid.apply(calc_pontos_row, axis=1, result_type="expand")
grid = pd.concat([grid, pts_df], axis=1)

def risco_to(total_pts: int) -> str:
    if total_pts > 20:
        return "NÃO"
    if total_pts >= 15:
        return "ATENÇÃO"
    return "SIM"

grid["RISCO DE TO"] = grid["TOTAL_PTS"].apply(lambda x: risco_to(int(x) if pd.notna(x) else 0))

# ✅ sobrescreve risco com STATUS (FERIAS/AFASTADO)
if "STATUS" in grid.columns:
    grid["STATUS"] = grid["STATUS"].fillna("").astype(str)
    grid.loc[grid["STATUS"].isin(["FERIAS", "AFASTADO"]), "RISCO DE TO"] = grid["STATUS"]

grid["MÊS"] = grid["MES"].apply(month_label)

# =========================================================
# SIDEBAR — FILTROS
# =========================================================
st.sidebar.header("Filtros")

ops_disp = sorted(base_master["OPERACAO"].dropna().astype(str).unique().tolist())
f_oper = st.sidebar.selectbox("Operação", ["Todos"] + ops_disp, index=0)

mes_opts = ["Todos"] + [month_label(m) for m in meses_all]
f_mes = st.sidebar.selectbox("Período", mes_opts, index=0)

f_func = st.sidebar.selectbox("Função", ["Todos"] + FUNCOES_PERMITIDAS, index=0)

ativs = sorted(base_master["ATIVIDADE"].dropna().astype(str).unique().tolist())
f_ativ = st.sidebar.selectbox("Atividade", ["Todos"] + ativs, index=0)

min_adm = pd.to_datetime(base_master["DATA_ADM_DT"], errors="coerce").min()
max_adm = pd.to_datetime(base_master["DATA_ADM_DT"], errors="coerce").max()
if pd.isna(min_adm) or pd.isna(max_adm):
    min_adm = pd.to_datetime("2024-01-01")
    max_adm = pd.to_datetime(datetime.today().date())

st.sidebar.subheader("Período de admissão")
col_ini, col_fim = st.sidebar.columns(2)
with col_ini:
    adm_ini = st.date_input(
        "Início",
        value=min_adm.date(),
        min_value=min_adm.date(),
        max_value=max_adm.date(),
        format="DD/MM/YYYY",
        key="raiox_adm_ini",
    )
with col_fim:
    adm_fim = st.date_input(
        "Fim",
        value=max_adm.date(),
        min_value=min_adm.date(),
        max_value=max_adm.date(),
        format="DD/MM/YYYY",
        key="raiox_adm_fim",
    )
if adm_ini > adm_fim:
    st.sidebar.warning("⚠️ Início maior que Fim. Ajustei automaticamente.")
    adm_ini, adm_fim = adm_fim, adm_ini

base_fil = base_master.copy()
if f_oper != "Todos":
    base_fil = base_fil[base_fil["OPERACAO"] == f_oper]
if f_func != "Todos":
    base_fil = base_fil[base_fil["FUNCAO"] == f_func]
if f_ativ != "Todos":
    base_fil = base_fil[base_fil["ATIVIDADE"].astype(str) == str(f_ativ)]

base_fil["DATA_ADM_DT"] = pd.to_datetime(base_fil["DATA_ADM_DT"], errors="coerce")
base_fil = base_fil[
    (base_fil["DATA_ADM_DT"].dt.date >= adm_ini) &
    (base_fil["DATA_ADM_DT"].dt.date <= adm_fim)
]

colabs_disp = sorted(base_fil["COLABORADOR"].dropna().astype(str).unique().tolist())
f_colab = st.sidebar.selectbox("Colaborador", ["Todos"] + colabs_disp, index=0)

# =========================================================
# APLICA FILTROS NA GRID
# =========================================================
df = grid.copy()

if f_oper != "Todos":
    df = df[df["OPERACAO"] == f_oper]
if f_func != "Todos":
    df = df[df["FUNCAO"] == f_func]
if f_ativ != "Todos":
    df = df[df["ATIVIDADE"].astype(str) == str(f_ativ)]

df["DATA_ADM_DT"] = pd.to_datetime(df["DATA_ADM_DT"], errors="coerce")
df = df[
    (df["DATA_ADM_DT"].dt.date >= adm_ini) &
    (df["DATA_ADM_DT"].dt.date <= adm_fim)
]

if f_mes != "Todos":
    mm, yy = f_mes.split("/")
    mes_key = f"{yy}-{mm}"
    df = df[df["MES"] == mes_key]

if f_colab != "Todos":
    nome_key = normalizar_nome(f_colab)
    df = df[df["NOME_KEY"] == nome_key]

# =========================================================
# BOX METAS/PONTUAÇÃO (lateral)
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("Pontuação")

op_meta_key = norm_operacao(f_oper) if f_oper != "Todos" else None
metas_op = METAS.get(op

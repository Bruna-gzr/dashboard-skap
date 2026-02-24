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
# TÍTULO
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

ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"
ARQ_SKAP   = DATA_DIR / "Skap.xlsx"
ARQ_PRONT  = DATA_DIR / "Prontuario Condutor.xlsx"
ARQ_VALES  = DATA_DIR / "Vales.xlsx"
ARQ_RV     = DATA_DIR / "RV.xlsx"
ARQ_ABS    = DATA_DIR / "Absenteismo.xlsx"
ARQ_ACID   = DATA_DIR / "Ocorrencia de acidentes.xlsx"
ARQ_DTO    = DATA_DIR / "Desvios de DTOs.xlsx"
ARQ_IND    = DATA_DIR / "Resultados indicadores operacionais.xlsx"

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

def nome_simple_from_key(key: str) -> str:
    """
    Fallback para match de nomes (remove partículas comuns).
    """
    key = normalizar_nome(key)
    tokens = [t for t in key.split() if t not in {"DE", "DA", "DO", "DAS", "DOS"}]
    return " ".join(tokens).strip()

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
    """
    Retorna em porcentagem (0-100).
    Aceita:
    - 0.811 -> 81.10%
    - "81,10%" / "81.10%" -> 81.10
    - 81.10 -> 81.10
    """
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
    """
    Retorna minutos como float.
    Aceita:
    - "00:30" (mm:ss)
    - "00:30:00" (hh:mm:ss)
    - número (já minutos)
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    s = s.replace("0 days ", "").replace("0 day ", "")

    v = pd.to_numeric(s.replace(",", "."), errors="coerce")
    if not pd.isna(v) and re.fullmatch(r"[-+]?\d+(\.\d+)?", s.replace(",", ".")) is not None:
        return float(v)

    parts = [p.strip() for p in s.split(":") if p.strip() != ""]
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
    return FUNCOES_MAP.get(normalizar_nome(cargo), "")

# =========================================================
# GRUPOS DE FUNÇÃO (define indicadores aplicáveis)
# =========================================================
FUNCOES_DISTRIB = {
    normalizar_nome("Ajudante de Distribuição"),
    normalizar_nome("Motorista de Distribuição"),
    normalizar_nome("Motorista de Van"),
}
FUNCOES_ARMAZEM = {
    normalizar_nome("Operador"),
    normalizar_nome("Ajudante de armazem"),
}

def grupo_funcao(funcao: str) -> str:
    fk = normalizar_nome(funcao)
    if fk in FUNCOES_ARMAZEM:
        return "ARMAZEM"
    if fk in FUNCOES_DISTRIB:
        return "DISTRIB"
    return "OUTROS"

# =========================================================
# METAS POR OPERAÇÃO (somente thresholds)
# Pontuação = 30 em todas as unidades via PONTOS_* por grupo
# =========================================================
METAS = {
    "CD CASCAVEL": {
        "PDV": {"meta": 3.5},
        "BEES": {"meta": 95.0},
        "TML": {"meta": 0.5},
        "JL":  {"meta": 80.0},
    },
    "CD DIADEMA": {
        "PDV": {"meta": 3.5},
        "BEES": {"meta": 95.0},
        "TML": {"meta": 0.5},
        "JL":  {"meta": 80.0},
    },
    "CD FOZ DO IGUACU": {
        "PDV": {"meta": 3.01},
        "BEES": {"meta": 95.0},
        "TML": {"meta": 0.5},
        "JL":  {"meta": 80.0},
    },
    "CD LONDRINA": {
        "PDV": {"meta": 3.5},
        "BEES": {"meta": 95.0},
        "TML": {"meta": 0.5},
        "JL":  {"meta": 80.0},
    },
    "CD LITORAL": {
        "PDV": {"meta": 3.5},
        "BEES": {"meta": 95.0},
        "TML": {"meta": 0.5},
        "JL":  {"meta": 80.0},
    },
    "CD SAO CRISTOVAO": {
        "PDV": {"meta": 3.5},
        "BEES": {"meta": 95.0},
        "TML": {"meta": 0.5},
        "JL":  {"meta": 80.0},
    },
    "CD FRANCISCO BELTRAO": {
        "PDV": {"meta": 2.84},
        "BEES": {"meta": 95.0},
        "TML": {"meta": 0.5},
        "JL":  {"meta": 89.0},
    },
}

METAS = {norm_operacao(k): v for k, v in METAS.items()}

# Pontos por grupo (sempre total 30)
PONTOS_DISTRIB = {
    "PDV": 5,
    "BEES": 1,
    "TML": 2,
    "JL": 3,
    "ABS": 5,
    "VALES": 5,
    "ACIDENTE": 5,
    "DTO": 4,
}  # soma 30

PONTOS_ARMAZEM = {
    "ABS": 10,
    "ACIDENTE": 10,
    "DTO": 5,
    "RV": 5,  # Média RV >= 80%
}  # soma 30

# =========================================================
# PRONTUÁRIO: ler "Empregado" + "Pontuação <br> Ponderada"
# =========================================================
def ler_prontuario_ponderada(path_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(path_xlsx, sheet_name=0)
    df = normalizar_colunas(df)

    # nome (prioriza EMPREGADO)
    col_nome = None
    for cand in ["EMPREGADO", "COLABORADOR", "NOME", "MOTORISTA", "CONDUTOR"]:
        if cand in df.columns:
            col_nome = cand
            break
    if col_nome is None:
        for c in df.columns:
            cc = normalizar_nome(c)
            if "EMPREG" in cc or "COLAB" in cc or "NOME" in cc:
                col_nome = c
                break
    if col_nome is None:
        return pd.DataFrame(columns=["NOME_KEY", "NOME_SIMPLE", "PRONT_PONDERADA"])

    df["NOME_KEY"] = df[col_nome].astype(str).map(normalizar_nome)
    df["NOME_SIMPLE"] = df["NOME_KEY"].map(nome_simple_from_key)

    # coluna ponderada (aceita variações com <br>)
    col_score = None
    for c in df.columns:
        cc = normalizar_nome(c)
        if ("PONTUACAO" in cc or "PONTUAC" in cc or "PONT" in cc) and "PONDERADA" in cc:
            col_score = c
            break
    if col_score is None:
        return pd.DataFrame(columns=["NOME_KEY", "NOME_SIMPLE", "PRONT_PONDERADA"])

    s = df[col_score].astype(str).str.replace(",", ".", regex=False)
    df["PRONT_PONDERADA"] = pd.to_numeric(s, errors="coerce")

    out = (
        df[["NOME_KEY", "NOME_SIMPLE", "PRONT_PONDERADA"]]
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
    skap   = pd.read_excel(ARQ_SKAP) if ARQ_SKAP.exists() else pd.DataFrame()
    vales  = pd.read_excel(ARQ_VALES) if ARQ_VALES.exists() else pd.DataFrame()
    rv     = pd.read_excel(ARQ_RV) if ARQ_RV.exists() else pd.DataFrame()
    abs_   = pd.read_excel(ARQ_ABS) if ARQ_ABS.exists() else pd.DataFrame()
    acid   = pd.read_excel(ARQ_ACID) if ARQ_ACID.exists() else pd.DataFrame()
    dto    = pd.read_excel(ARQ_DTO) if ARQ_DTO.exists() else pd.DataFrame()
    ind    = pd.read_excel(ARQ_IND) if ARQ_IND.exists() else pd.DataFrame()

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
ativos["NOME_SIMPLE"] = ativos["NOME_KEY"].map(nome_simple_from_key)
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
    skap["NOME_SIMPLE"] = skap["NOME_KEY"].map(nome_simple_from_key)
    skap_niveis = skap[["NOME_KEY", "NOME_SIMPLE", "NIVEIS"]].drop_duplicates(subset=["NOME_KEY"], keep="last")
else:
    skap_niveis = pd.DataFrame(columns=["NOME_KEY", "NOME_SIMPLE", "NIVEIS"])

# =========================================================
# PRONTUÁRIO
# =========================================================
if pront is None or pront.empty:
    pront = pd.DataFrame(columns=["NOME_KEY", "NOME_SIMPLE", "PRONT_PONDERADA"])
else:
    pront = normalizar_colunas(pront)
    for c in ["NOME_KEY", "NOME_SIMPLE", "PRONT_PONDERADA"]:
        if c not in pront.columns:
            pront[c] = np.nan if c == "PRONT_PONDERADA" else ""

# =========================================================
# BASE MASTER (merge com fallback de nome_simple)
# =========================================================
base_master = (
    ativos
    .merge(skap_niveis[["NOME_KEY", "NIVEIS"]].drop_duplicates("NOME_KEY"), on="NOME_KEY", how="left")
)

# merge pront por NOME_KEY
base_master = base_master.merge(
    pront[["NOME_KEY", "PRONT_PONDERADA"]].drop_duplicates("NOME_KEY"),
    on="NOME_KEY",
    how="left"
)

# fallback: se PRONT_PONDERADA vazio, tenta por NOME_SIMPLE
if "PRONT_PONDERADA" in base_master.columns:
    missing = base_master["PRONT_PONDERADA"].isna()
    if missing.any():
        base_master_f = base_master.loc[missing, ["NOME_SIMPLE"]].copy()
        pront_f = pront[["NOME_SIMPLE", "PRONT_PONDERADA"]].dropna(subset=["NOME_SIMPLE"]).drop_duplicates("NOME_SIMPLE", keep="last")
        base_master_f = base_master_f.merge(pront_f, on="NOME_SIMPLE", how="left")
        base_master.loc[missing, "PRONT_PONDERADA"] = base_master_f["PRONT_PONDERADA"].values

if "NIVEIS" not in base_master.columns:
    base_master["NIVEIS"] = ""

# Prontuário só para motoristas
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
# EVENTOS / OCORRÊNCIAS POR MÊS
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

# RV (por mês)  -> vamos usar PCT_RV (média do % no período/mês)
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
ind_m = pd.DataFrame(columns=["NOME_KEY", "MES", "PDV", "BEES", "TML_MIN", "JL", "STATUS", "STATUS_RAW"])
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
    ind["STATUS_RAW"] = ind["STATUS"].astype(str)

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

    def pick_status_raw(series: pd.Series) -> str:
        vals = [str(x).strip() for x in series.dropna().astype(str).tolist() if str(x).strip() != ""]
        return vals[-1] if vals else ""

    ind_m = (
        ind.groupby(["NOME_KEY", "MES"], dropna=False)
        .agg(
            PDV=("PDV_NUM", "mean"),
            BEES=("BEES_NUM", "mean"),
            TML_MIN=("TML_MIN", "mean"),
            JL=("JL_NUM", "mean"),
            STATUS=("STATUS_NORM", pick_status),
            STATUS_RAW=("STATUS_RAW", pick_status_raw),
        )
        .reset_index()
    )

# Contagens por mês
vales_m = vales_ev.groupby(["NOME_KEY", "MES"]).size().reset_index(name="VALES")
acid_m = acid_ev.groupby(["NOME_KEY", "MES"]).size().reset_index(name="ACIDENTE")
dto_m  = dto_ev.groupby(["NOME_KEY", "MES"]).size().reset_index(name="DTO")
abs_m  = abs_ev.groupby(["NOME_KEY", "MES"]).size().reset_index(name="ABS")

# =========================================================
# GRID colaborador x mês
# =========================================================
meses_sets = []
for df_ in [vales_m, acid_m, dto_m, abs_m, rv_m, ind_m]:
    if df_ is not None and not df_.empty and "MES" in df_.columns:
        meses_sets.append(df_["MES"].dropna().unique().tolist())
meses_all = sorted(list(set([m for sub in meses_sets for m in sub])))

if not meses_all:
    meses_all = [pd.Timestamp.today().to_period("M").astype(str)]

colab_univ = base_master[[
    "NOME_KEY", "NOME_SIMPLE", "COLABORADOR", "OPERACAO", "OPER_KEY", "ATIVIDADE",
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

grid = left_join(grid, ind_m, ["PDV", "BEES", "TML_MIN", "JL", "STATUS", "STATUS_RAW"])
grid = left_join(grid, abs_m, ["ABS"])
grid = left_join(grid, vales_m, ["VALES"])
grid = left_join(grid, acid_m, ["ACIDENTE"])
grid = left_join(grid, dto_m, ["DTO"])
grid = left_join(grid, rv_m, ["TOTAL_RV", "PCT_RV", "RECARGA"])

for c in ["ABS", "VALES", "ACIDENTE", "DTO"]:
    if c in grid.columns:
        grid[c] = pd.to_numeric(grid[c], errors="coerce").fillna(0).astype(int)

# remover meses anteriores à admissão
grid["_MES_DT"] = pd.to_datetime(grid["MES"] + "-01", errors="coerce")
grid["_ADM_MES_DT"] = pd.to_datetime(grid["DATA_ADM_DT"], errors="coerce").dt.to_period("M").dt.to_timestamp()
grid = grid[grid["_MES_DT"] >= grid["_ADM_MES_DT"]].copy()
grid = grid.drop(columns=["_MES_DT", "_ADM_MES_DT"])

grid["MÊS"] = grid["MES"].apply(month_label)

# =========================================================
# PONTUAÇÃO POR GRUPO (sempre total 30)
# =========================================================
def calc_pontos_row(row) -> dict:
    op_key = norm_operacao(row.get("OPERACAO", ""))
    metas_op = METAS.get(op_key, {})

    grp = grupo_funcao(row.get("FUNCAO", ""))

    res = {
        "PTS_PDV": 0, "PTS_BEES": 0, "PTS_TML": 0, "PTS_JL": 0,
        "PTS_ABS": 0, "PTS_VALES": 0, "PTS_ACIDENTE": 0, "PTS_DTO": 0,
        "PTS_RV": 0,
        "TOTAL_PTS": 0
    }

    # DISTRIB (todos indicadores)
    if grp in ["DISTRIB", "OUTROS"]:
        # metas
        v_pdv = row.get("PDV", np.nan)
        v_bees = row.get("BEES", np.nan)
        v_tml = row.get("TML_MIN", np.nan)
        v_jl = row.get("JL", np.nan)

        if pd.notna(v_pdv) and "PDV" in metas_op and v_pdv <= metas_op["PDV"]["meta"]:
            res["PTS_PDV"] = PONTOS_DISTRIB["PDV"]
        if pd.notna(v_bees) and "BEES" in metas_op and v_bees >= metas_op["BEES"]["meta"]:
            res["PTS_BEES"] = PONTOS_DISTRIB["BEES"]
        if pd.notna(v_tml) and "TML" in metas_op and v_tml <= metas_op["TML"]["meta"]:
            res["PTS_TML"] = PONTOS_DISTRIB["TML"]
        if pd.notna(v_jl) and "JL" in metas_op and v_jl >= metas_op["JL"]["meta"]:
            res["PTS_JL"] = PONTOS_DISTRIB["JL"]

        if int(row.get("ABS", 0)) == 0:
            res["PTS_ABS"] = PONTOS_DISTRIB["ABS"]
        if int(row.get("VALES", 0)) == 0:
            res["PTS_VALES"] = PONTOS_DISTRIB["VALES"]
        if int(row.get("ACIDENTE", 0)) == 0:
            res["PTS_ACIDENTE"] = PONTOS_DISTRIB["ACIDENTE"]
        if int(row.get("DTO", 0)) == 0:
            res["PTS_DTO"] = PONTOS_DISTRIB["DTO"]

        res["TOTAL_PTS"] = (
            res["PTS_PDV"] + res["PTS_BEES"] + res["PTS_TML"] + res["PTS_JL"] +
            res["PTS_ABS"] + res["PTS_VALES"] + res["PTS_ACIDENTE"] + res["PTS_DTO"]
        )
        return res

    # ARMAZEM (ABS/ACIDENTE/DTO/RV)
    if grp == "ARMAZEM":
        if int(row.get("ABS", 0)) == 0:
            res["PTS_ABS"] = PONTOS_ARMAZEM["ABS"]
        if int(row.get("ACIDENTE", 0)) == 0:
            res["PTS_ACIDENTE"] = PONTOS_ARMAZEM["ACIDENTE"]
        if int(row.get("DTO", 0)) == 0:
            res["PTS_DTO"] = PONTOS_ARMAZEM["DTO"]

        v_rv = row.get("PCT_RV", np.nan)
        if pd.notna(v_rv) and float(v_rv) >= 80.0:
            res["PTS_RV"] = PONTOS_ARMAZEM["RV"]

        res["TOTAL_PTS"] = res["PTS_ABS"] + res["PTS_ACIDENTE"] + res["PTS_DTO"] + res["PTS_RV"]
        return res

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

# sobrescreve risco com STATUS (FERIAS/AFASTADO)
if "STATUS" in grid.columns:
    grid["STATUS"] = grid["STATUS"].fillna("").astype(str)
    grid.loc[grid["STATUS"].isin(["FERIAS", "AFASTADO"]), "RISCO DE TO"] = grid["STATUS"]

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

# Colaborador depende dos filtros acima
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
# Define VISUALIZAÇÃO por função selecionada
# =========================================================
if f_func == "Todos":
    view_mode = "ALL"
else:
    fk = normalizar_nome(f_func)
    if fk in FUNCOES_ARMAZEM:
        view_mode = "ARMAZEM"
    else:
        view_mode = "DISTRIB"

# =========================================================
# BOX METAS/PONTUAÇÃO (sidebar) -> total 30
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("Pontuação (total possível: 30 pts)")

def linha_meta(nome, meta, pts):
    st.sidebar.markdown(f"- **{nome}**: meta **{meta}** · **{pts} pts**")

op_meta_key = norm_operacao(f_oper) if f_oper != "Todos" else None
metas_op = METAS.get(op_meta_key, None) if op_meta_key else None

if metas_op is None:
    st.sidebar.caption("Selecione uma operação para ver metas.")
else:
    if view_mode in ["ALL", "DISTRIB"]:
        st.sidebar.markdown("**Distribuição (Ajudante/Motoristas)**")
        linha_meta("DEV PDV", f"≤ {metas_op['PDV']['meta']}%", PONTOS_DISTRIB["PDV"])
        linha_meta("BEES",    f"≥ {metas_op['BEES']['meta']}%", PONTOS_DISTRIB["BEES"])
        linha_meta("TML",     "≤ 00:30", PONTOS_DISTRIB["TML"])
        linha_meta("JL",      f"≥ {metas_op['JL']['meta']}%", PONTOS_DISTRIB["JL"])
        linha_meta("ABS",     "0", PONTOS_DISTRIB["ABS"])
        linha_meta("VALES",   "0", PONTOS_DISTRIB["VALES"])
        linha_meta("ACIDENTE","0", PONTOS_DISTRIB["ACIDENTE"])
        linha_meta("Desvios DTO","0", PONTOS_DISTRIB["DTO"])

    if view_mode in ["ALL", "ARMAZEM"]:
        st.sidebar.markdown("**Armazém (Operador/Ajudante armazém)**")
        linha_meta("ABS",        "0", PONTOS_ARMAZEM["ABS"])
        linha_meta("ACIDENTE",   "0", PONTOS_ARMAZEM["ACIDENTE"])
        linha_meta("Desvios DTO","0", PONTOS_ARMAZEM["DTO"])
        linha_meta("Média RV",   "≥ 80%", PONTOS_ARMAZEM["RV"])

# =========================================================
# REFERÊNCIAS
# - ref_last_colab: só existe quando há colaborador selecionado
# - ref_avg_all: médias do filtro atual (indicadores à esquerda)
# =========================================================
ref_last_colab = None
ref_avg_all = None

if not df.empty:
    ref_avg_all = {
        "JL": df["JL"].mean(),
        "PDV": df["PDV"].mean(),
        "BEES": df["BEES"].mean(),
        "TML_MIN": df["TML_MIN"].mean(),
        "ABS": df["ABS"].mean(),
        "VALES": df["VALES"].mean(),
        "ACIDENTE": df["ACIDENTE"].mean(),
        "DTO": df["DTO"].mean(),
        "PCT_RV": df["PCT_RV"].mean(),
    }

    if f_colab != "Todos":
        ref_last_colab = df.sort_values("MES", ascending=False).iloc[0].copy()

# =========================================================
# ESTILOS
# =========================================================
def color_risco(v):
    v = str(v)
    if v == "NÃO":
        return "color: #22c55e; font-weight:900;"
    if v == "ATENÇÃO":
        return "color: #facc15; font-weight:900;"
    if v == "SIM":
        return "color: #ef4444; font-weight:900;"
    if v in ["FERIAS", "AFASTADO"]:
        return "color: #94a3b8; font-weight:900;"
    return ""

def color_pts_zero(v):
    try:
        vv = float(v)
    except Exception:
        vv = 0.0
    if vv > 0:
        return "color: #22c55e; font-weight:900;"
    return "color: #ef4444; font-weight:900;"

def color_rv(v):
    try:
        vv = float(v)
    except Exception:
        return ""
    if pd.isna(vv):
        return ""
    if vv >= 80.0:
        return "color: #22c55e; font-weight:900;"
    return "color: #ef4444; font-weight:900;"

def risco_color_hex(risco: str) -> str:
    risco = str(risco)
    if risco == "NÃO":
        return "#22c55e"
    if risco == "ATENÇÃO":
        return "#facc15"
    if risco == "SIM":
        return "#ef4444"
    return "#94a3b8"

# =========================================================
# LAYOUT
# =========================================================
left, right = st.columns([1.05, 2.95], gap="large")

with left:
    st.markdown("#### Colaborador")

    # quando não tem colaborador selecionado, fica vazio
    nome_card = ref_last_colab["COLABORADOR"] if ref_last_colab is not None else ""
    subt_card = ""
    if ref_last_colab is not None:
        subt_card = f"{ref_last_colab['OPERACAO']} · {ref_last_colab['FUNCAO']}"

    st.markdown(
        f"""
        <div style='padding:12px;border-radius:14px;background:#22262f; min-height:64px;'>
          <div style='font-weight:900;font-size:18px;line-height:1.1;'>
            {nome_card}
          </div>
          <div style='opacity:.82;font-size:12px;margin-top:4px;'>
            {subt_card}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Indicadores")

    # fonte maior e sem PTS
    def linha_ind(icon, nome, resultado):
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:12px 12px;border-radius:12px;background:#22262f;margin-bottom:8px;">
              <div style="display:flex;align-items:center;gap:10px;">
                <div style="font-size:20px;opacity:.9;">{icon}</div>
                <div>
                  <div style="font-weight:900;font-size:15px;opacity:.98;">{nome}</div>
                  <div style="font-size:14px;opacity:.85;">Resultado (média): <b>{resultado}</b></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if ref_avg_all is None:
        st.info("Sem dados para os filtros atuais.")
    else:
        # resultados
        jl_res   = fmt_pct(ref_avg_all.get("JL", np.nan)) or "-"
        pdv_res  = fmt_pct(ref_avg_all.get("PDV", np.nan)) or "-"
        bees_res = fmt_pct(ref_avg_all.get("BEES", np.nan)) or "-"
        tml_res  = fmt_min(ref_avg_all.get("TML_MIN", np.nan)) or "-"
        rv_res   = fmt_pct(ref_avg_all.get("PCT_RV", np.nan)) or "-"

        if view_mode in ["ALL", "DISTRIB"]:
            linha_ind("👥", "JL", jl_res)
            linha_ind("📦", "DEV PDV", pdv_res)
            linha_ind("🟨", "BEES", bees_res)
            linha_ind("⏱️", "TML", tml_res)
            linha_ind("🩹", "ABS", f"{ref_avg_all.get('ABS', 0):.2f}")
            linha_ind("🎫", "VALES", f"{ref_avg_all.get('VALES', 0):.2f}")
            linha_ind("⚠️", "ACIDENTE", f"{ref_avg_all.get('ACIDENTE', 0):.2f}")
            linha_ind("📄", "Desvios DTO", f"{ref_avg_all.get('DTO', 0):.2f}")
            linha_ind("📈", "Média RV", rv_res)

        if view_mode == "ARMAZEM":
            linha_ind("🩹", "ABS", f"{ref_avg_all.get('ABS', 0):.2f}")
            linha_ind("⚠️", "ACIDENTE", f"{ref_avg_all.get('ACIDENTE', 0):.2f}")
            linha_ind("📄", "Desvios DTO", f"{ref_avg_all.get('DTO', 0):.2f}")
            linha_ind("📈", "Média RV", rv_res)

    # TOP 10 + cargo
    st.markdown("### 🏆 TOP 10")
    if df.empty:
        st.info("Sem dados para ranking com os filtros atuais.")
    else:
        rank = (
            df.groupby(["NOME_KEY", "COLABORADOR", "FUNCAO"], dropna=False)
            .agg(MEDIA_TOTAL=("TOTAL_PTS", "mean"))
            .reset_index()
            .sort_values("MEDIA_TOTAL", ascending=False)
            .head(10)
        )
        rank["MEDIA_TOTAL"] = rank["MEDIA_TOTAL"].map(lambda x: f"{x:.1f}")
        rank_out = rank[["COLABORADOR", "FUNCAO", "MEDIA_TOTAL"]].rename(columns={"FUNCAO": "CARGO"})
        st.dataframe(centralizar_tabela(rank_out), use_container_width=True, height=360)

with right:
    # Top cards
    top1, top2, top3 = st.columns([2.4, 1.3, 1.3])

    with top1:
        st.markdown(
            f"<h4 style='margin:0;'> {ref_last_colab['FUNCAO'] if ref_last_colab is not None else ''} </h4>",
            unsafe_allow_html=True
        )

    with top2:
        if ref_last_colab is None or pd.isna(ref_last_colab.get("DATA_ADM_DT", None)):
            st.metric("Tempo de casa (meses)", "-")
        else:
            meses = int((pd.Timestamp.today().normalize() - pd.to_datetime(ref_last_colab["DATA_ADM_DT"]).normalize()).days / 30.44)
            st.metric("Tempo de casa (meses)", f"{meses}")

    with top3:
        st.metric("Admissão", (ref_last_colab["ADMISSAO"] if ref_last_colab is not None else "-"))

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Card Risco do último mês com cores
    if ref_last_colab is None:
        st.info("Selecione um colaborador para ver o risco do último mês.")
    else:
        risco_ultimo = str(ref_last_colab.get("RISCO DE TO", "-"))
        cor = risco_color_hex(risco_ultimo)
        st.markdown(
            f"""
            <div style="padding:14px;border-radius:14px;background:#22262f;margin-bottom:14px;">
              <div style="font-size:12px;opacity:.75;">RISCO DE TO? (último mês)</div>
              <div style="font-size:22px;font-weight:900;margin-top:4px;color:{cor};">{risco_ultimo}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Cards em caixa
    with st.container(border=True):
        cA, cB, cC, cD = st.columns(4)

        with cA:
            st.metric("SKAP - Níveis", (ref_last_colab["NIVEIS"] if ref_last_colab is not None and str(ref_last_colab.get("NIVEIS","")).strip() != "" else "-"))

        with cB:
            v = ref_last_colab.get("PRONT_PONDERADA", np.nan) if ref_last_colab is not None else np.nan
            faixa = ref_last_colab.get("PRONT_FAIXA", "-") if ref_last_colab is not None else "-"
            cor = ref_last_colab.get("PRONT_COR", "#94a3b8") if ref_last_colab is not None else "#94a3b8"
            if pd.isna(v):
                st.metric("Prontuário", "-")
            else:
                st.markdown(
                    f"""
                    <div style="padding:10px;border-radius:12px;background:#2b2f38;">
                      <div style="font-weight:700;margin-bottom:4px;">Prontuário</div>
                      <div style="font-weight:900;color:{cor};">{faixa}</div>
                      <div style="opacity:0.9;font-size:12px;">Ponderada: <b>{v:.3f}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with cC:
            col = "CICLO DE GENTE 2025"
            if col in base_master.columns and ref_last_colab is not None:
                st.metric("Ciclo de gente", str(ref_last_colab.get(col, "-")) or "-")
            else:
                st.metric("Ciclo de gente", "-")

        with cD:
            st.metric("Pontuação Total (média)", "-" if df.empty else f"{df['TOTAL_PTS'].mean():.1f}")

    st.divider()

    # Gráfico
    st.subheader("📊 Pontuação x mês")

    graf = df.copy()
    if graf.empty:
        st.info("Sem dados para o gráfico com os filtros atuais.")
    else:
        g = (
            graf.groupby(["MES"], dropna=False)["TOTAL_PTS"]
            .mean()
            .reset_index()
            .sort_values("MES")
        )
        g["MÊS"] = g["MES"].apply(month_label)

        fig = px.bar(g, x="MÊS", y="TOTAL_PTS", text=g["TOTAL_PTS"].map(lambda x: f"{x:.0f}"))
        fig.update_layout(yaxis_title="Pontuação (TOTAL)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Tabela
    st.subheader("📋 Tabela de Pontuação (por colaborador e mês)")

    if df.empty:
        st.info("Sem dados para tabela com os filtros atuais.")
    else:
        t = df.copy()

        # formatar indicadores
        t["DEV PDV"] = t["PDV"].apply(fmt_pct)
        t["BEES"]    = t["BEES"].apply(fmt_pct)
        t["TML"]     = t["TML_MIN"].apply(fmt_min)
        t["JL"]      = t.get("JL", np.nan).apply(fmt_pct)
        t["MEDIA_RV"] = t.get("PCT_RV", np.nan)  # numérico 0-100
        t["MÉDIA RV"] = t["MEDIA_RV"].apply(fmt_pct)

        t["ABS"]     = t["ABS"].astype(int)
        t["VALES"]   = t["VALES"].astype(int)
        t["ACIDENTE"]= t["ACIDENTE"].astype(int)
        t["DTO"]     = t["DTO"].astype(int)

        # STATUS exibido após CARGO; DESCLASSIFICADO -> TRABALHANDO
        def status_exibir(x):
            s = "" if pd.isna(x) else str(x).strip()
            if normalizar_nome(s) == normalizar_nome("DESCLASSIFICADO"):
                return "TRABALHANDO"
            return s

        t["STATUS_EXIB"] = t.get("STATUS_RAW", "").apply(status_exibir)

        # Base da tabela (com nomenclaturas novas)
        out = pd.DataFrame({
            ("MÊS",""): t["MÊS"],
            ("COLABORADOR",""): t["COLABORADOR"],
            ("CARGO",""): t["FUNCAO"],
            ("STATUS",""): t["STATUS_EXIB"],
            ("RISCO DE TO?",""): t["RISCO DE TO"],
            ("TOTAL",""): t["TOTAL_PTS"],
        })

        def add_indicador(nome, col_res, col_pts=None):
            out[(nome, "Resultado")] = t[col_res]
            if col_pts is not None:
                out[(nome, "PTS")] = t[col_pts]

        # Define quais indicadores entram na tabela conforme view_mode
        if view_mode in ["ALL", "DISTRIB"]:
            add_indicador("JL",       "JL",       "PTS_JL")
            add_indicador("DEV PDV",  "DEV PDV",  "PTS_PDV")
            add_indicador("BEES",     "BEES",     "PTS_BEES")
            add_indicador("TML",      "TML",      "PTS_TML")
            add_indicador("ABS",      "ABS",      "PTS_ABS")
            add_indicador("VALES",    "VALES",    "PTS_VALES")
            add_indicador("ACIDENTE", "ACIDENTE", "PTS_ACIDENTE")
            add_indicador("Desvios DTO", "DTO",   "PTS_DTO")
            # ✅ Média RV após DTO (sem PTS para Distribuição; somente resultado)
            out[("Média RV", "Resultado")] = t["MÉDIA RV"]

        if view_mode == "ARMAZEM":
            add_indicador("ABS",      "ABS",      "PTS_ABS")
            add_indicador("ACIDENTE", "ACIDENTE", "PTS_ACIDENTE")
            add_indicador("Desvios DTO", "DTO",   "PTS_DTO")
            out[("Média RV", "Resultado")] = t["MÉDIA RV"]
            out[("Média RV", "PTS")] = t["PTS_RV"]

        # ordenar por mês/colaborador
        out[("_MESKEY","")] = pd.to_datetime(t["MES"] + "-01", errors="coerce")
        out = out.sort_values([("_MESKEY",""), ("COLABORADOR","")]).drop(columns=[("_MESKEY","")])

        # estilos: centralizado + risco colorido + TOTAL bold + PTS verde/vermelho + RV verde/vermelho
        sty = (
            out.style
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
        )

        pts_cols = [c for c in out.columns if isinstance(c, tuple) and c[1] == "PTS"]
        if pts_cols:
            sty = sty.applymap(color_pts_zero, subset=pts_cols)

        # Risco
        sty = sty.applymap(color_risco, subset=[("RISCO DE TO?","")])

        # TOTAL em negrito
        sty = sty.applymap(lambda v: "font-weight:900;" if isinstance(v, (int, float, np.integer, np.floating)) else "", subset=[("TOTAL","")])

        # Média RV (resultado) colorido por >=80
        if ("Média RV", "Resultado") in out.columns:
            sty = sty.applymap(lambda v: color_rv(parse_percent(v)), subset=[("Média RV", "Resultado")])

        st.dataframe(sty, use_container_width=True, height=520)

        # Export
        export_df = out.copy()
        export_df.columns = [c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in export_df.columns]
        excel = preparar_excel_para_download(export_df, sheet_name="RAIO_X")
        st.download_button(
            "⬇️ Baixar Excel (Tabela RAIO X)",
            data=excel,
            file_name="raio_x_operacao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# =========================================================
# NOTAS
# =========================================================
with st.expander("ℹ️ Notas rápidas"):
    st.write(
        "- Match entre planilhas é feito por **nome normalizado** e com **fallback** removendo partículas (de/da/do/dos/das).\n"
        "- Prontuário: somente para **Motorista de Distribuição** e **Motorista de Van**, usando **Pontuação Ponderada**.\n"
        "- Meses anteriores à admissão **não aparecem**.\n"
        "- Indicadores por mês: quando há mais de um registro no mês, o app usa **média** (PDV/BEES/TML/JL e RV%).\n"
        "- Se STATUS do indicador for **FERIAS** ou **AFASTADO/AFS**, a coluna **RISCO DE TO** mostra esse status.\n"
        "- Na tabela, **DESCLASSIFICADO** é exibido como **TRABALHANDO**."
    )

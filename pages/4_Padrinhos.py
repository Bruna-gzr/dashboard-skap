# pages/4_Padrinhos.py
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Importar funções do app.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import aplicar_filtro, get_operacao

st.set_page_config(page_title="Gestão de Padrinhos", layout="wide")

# Pega a operação do usuário logado
OPERACAO_USUARIO = get_operacao()

st.title("👨🏻‍🎓 Gestão de Padrinhos")

if OPERACAO_USUARIO != "Todas":
    st.caption(f"📍 Operação: **{OPERACAO_USUARIO}**")
else:
    st.caption("📍 Visualizando TODAS as operações")

# Botão para voltar
if st.button("← Voltar ao Menu"):
    st.switch_page("app.py")

# =========================
# RESTO DO SEU CÓDIGO (mantido igual, apenas ajustando os filtros)
# =========================

import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
import plotly.express as px

# =========================
# Imports opcionais p/ fuzzy (com fallback otimizado)
# =========================
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# =========================
# Estilo
# =========================
st.markdown("""
<style>
.stApp { background: #0b0b0b; color: #f5f5f5; }
h1 { color: #f0d36b !important; }
h2, h3 { color: white !important; }

.card {
  background: #000000;
  border-radius: 18px;
  padding: 16px 18px;
  border: 1px solid #222;
  box-shadow: 0 6px 20px rgba(0,0,0,0.35);
  margin-bottom: 14px;
}

.small-muted { color: #bdbdbd; font-size: 0.9rem; }
hr { border: none; border-top: 1px solid #222; }

section[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #222;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #f5f5f5 !important;
}

div[data-baseweb="select"] > div {
    background-color: #10182b !important;
    border: 1px solid #1e2a44 !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] > div {
    background-color: #10182b !important;
    border: 1px solid #1e2a44 !important;
    border-radius: 10px !important;
}

[data-testid="stDateInput"] > div {
    background-color: #10182b !important;
    border-radius: 10px !important;
}

[data-testid="metric-container"] {
    background: #111111;
    border: 1px solid #222;
    padding: 10px 14px;
    border-radius: 14px;
}

.info-box {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 0.92rem;
    color: #d9d9d9;
    margin-bottom: 10px;
}

.lista-box {
    background: #1e1e1e;
    border: 1px solid #343434;
    border-radius: 10px;
    padding: 0;
    overflow: hidden;
    margin-bottom: 12px;
}

.lista-titulo {
    background: #2b2b2b;
    color: #ffd966;
    font-weight: 700;
    padding: 8px 10px;
    border-bottom: 1px solid #3a3a3a;
}

.lista-scroll {
    height: 520px;
    overflow-y: auto;
    overflow-x: hidden;
    background: #1f1f1f;
}

.lista-scroll::-webkit-scrollbar {
    width: 8px;
}

.lista-scroll::-webkit-scrollbar-track {
    background: #202020;
}

.lista-scroll::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 8px;
}

.lista-linha {
    padding: 7px 10px;
    border-bottom: 1px solid #303030;
    color: #ffffff;
    font-size: 0.92rem;
    line-height: 1.35;
    word-break: break-word;
    min-height: 30px;
}

.lista-linha:nth-child(even) {
    background: #262626;
}

.titulo-branco {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.15rem;
    margin-top: 8px;
    margin-bottom: 10px;
}

.titulo-amarelo {
    color: #f0d36b !important;
    font-weight: 700;
    font-size: 1.15rem;
    margin-top: 8px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================

@st.cache_data(ttl=3600)
def cached_norm_text(text: str) -> str:
    if pd.isna(text) or text == "":
        return ""
    s = str(text).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return cached_norm_text(x)

def norm_text_nome_flex(x) -> str:
    s = norm_text(x)
    if not s:
        return ""
    tokens = [t for t in s.split() if len(t) > 1]
    return " ".join(tokens)

def clean_cpf(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\D", "", s)
    if len(s) == 10:
        s = "0" + s
    return s if len(s) == 11 else ""

def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100

def token_overlap(a: str, b: str) -> int:
    sa = set(a.split())
    sb = set(b.split())
    return len(sa.intersection(sb))

def primeiro_nome(a: str) -> str:
    toks = a.split()
    return toks[0] if toks else ""

def ult_nome(a: str) -> str:
    toks = a.split()
    return toks[-1] if toks else ""

def parse_horario_texto(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3)) if m.group(3) else 0
    if hh > 23 or mm > 59 or ss > 59:
        return None
    return hh, mm, ss

def combinar_data_hora(df: pd.DataFrame, col_data="Data Cadastro", col_hora="Horário da resposta") -> pd.Series:
    data = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True) if col_data in df.columns else pd.Series(pd.NaT, index=df.index)
    hora = df[col_hora] if col_hora in df.columns else pd.Series(index=df.index, dtype="object")
    valores = []
    for d, h in zip(data, hora):
        if pd.isna(d):
            valores.append(pd.NaT)
            continue
        parsed = parse_horario_texto(h)
        if parsed is None:
            valores.append(pd.Timestamp(d).normalize())
        else:
            hh, mm, ss = parsed
            valores.append(pd.Timestamp(d).normalize() + pd.Timedelta(hours=hh, minutes=mm, seconds=ss))
    return pd.to_datetime(pd.Series(valores, index=df.index), errors="coerce")

from functools import lru_cache

@lru_cache(maxsize=10000)
def similaridade_nome_cached(resp_nome: str, cand_nome: str) -> float:
    if not resp_nome or not cand_nome:
        return 0
    resp_nome_flex = norm_text_nome_flex(resp_nome)
    cand_nome_flex = norm_text_nome_flex(cand_nome)
    if not resp_nome_flex or not cand_nome_flex:
        return 0
    if RAPIDFUZZ_OK:
        s1 = fuzz.token_set_ratio(resp_nome_flex, cand_nome_flex)
        s2 = fuzz.token_sort_ratio(resp_nome_flex, cand_nome_flex)
        s3 = fuzz.WRatio(resp_nome_flex, cand_nome_flex)
        s4 = fuzz.partial_ratio(resp_nome_flex, cand_nome_flex)
        seq = sequence_ratio(resp_nome_flex, cand_nome_flex)
        score = max(s1, s2 * 0.99, s3 * 0.98, s4 * 0.97, seq * 0.96)
        inter = token_overlap(resp_nome_flex, cand_nome_flex)
        if inter >= 2:
            score += 2.5
        if inter >= 3:
            score += 1.5
        if primeiro_nome(resp_nome_flex) and primeiro_nome(resp_nome_flex) == primeiro_nome(cand_nome_flex):
            score += 2.5
        if ult_nome(resp_nome_flex) and ult_nome(resp_nome_flex) == ult_nome(cand_nome_flex):
            score += 2.0
        return min(score, 100)
    base = max(
        sequence_ratio(resp_nome_flex, cand_nome_flex),
        sequence_ratio(" ".join(sorted(resp_nome_flex.split())), " ".join(sorted(cand_nome_flex.split())))
    )
    inter = token_overlap(resp_nome_flex, cand_nome_flex)
    if inter >= 2:
        base += 3
    if inter >= 3:
        base += 2
    if primeiro_nome(resp_nome_flex) == primeiro_nome(cand_nome_flex):
        base += 2
    if ult_nome(resp_nome_flex) == ult_nome(cand_nome_flex):
        base += 2
    return min(base, 100)

def similaridade_nome(resp_nome: str, cand_nome: str) -> float:
    return similaridade_nome_cached(resp_nome, cand_nome)

def classificar_faixa_aderencia(valor):
    if pd.isna(valor):
        return "< 80%"
    if valor >= 90:
        return ">= 90%"
    if valor >= 80:
        return "80% a 89%"
    return "< 80%"

def cor_status(s):
    if s == "Realizado no prazo":
        return "background-color: #2e7d32; color: white;"
    if s == "Realizado antes do prazo":
        return "background-color: #1b5e20; color: white;"
    if s == "Realizado fora do prazo":
        return "background-color: #ef6c00; color: black;"
    if s == "Não realizado - Atenção":
        return "background-color: #f0d36b; color: black;"
    if s == "Não realizado - Fora do prazo":
        return "background-color: #c62828; color: white;"
    return ""

def style_table(df):
    if "Status" in df.columns:
        return df.style.map(cor_status, subset=["Status"]).set_properties(**{"text-align": "center"})
    return df

@st.cache_data(show_spinner=True, ttl=3600)
def carregar_excel_primeira_aba(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    xls = pd.ExcelFile(path)
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

def obter_ultima_atualizacao_arquivos(paths: list[Path]) -> pd.Timestamp | None:
    datas = []
    for p in paths:
        if p.exists():
            datas.append(pd.to_datetime(datetime.fromtimestamp(p.stat().st_mtime)))
    if not datas:
        return None
    return max(datas)

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "dados") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

def formatar_datas_para_tabela(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Data Admissão", "Prazo Mín", "Prazo Máx", "Data Realização", "Data Cadastro"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", dayfirst=True).dt.strftime("%d/%m/%Y")
    return out

def renomear_colunas_duplicadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = pd.Series(df.columns, dtype="object")
    contagem = {}
    novas = []
    for c in cols:
        if c not in contagem:
            contagem[c] = 0
            novas.append(c)
        else:
            contagem[c] += 1
            novas.append(f"{c}__{contagem[c]}")
    df.columns = novas
    return df

# =========================
# Paths
# =========================
DATA_DIR = Path(__file__).parent.parent / "data"
ARQ_ADMITIDOS = DATA_DIR / "Admitidos.xlsx"
ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"
ARQ_NPS = DATA_DIR / "NPS Mentor.xlsx"
ARQ_BATEPAPO = DATA_DIR / "Bate papo mentor.xlsx"

# =========================
# Carregamento
# =========================
try:
    admitidos = carregar_excel_primeira_aba(ARQ_ADMITIDOS)
    # 🔴 FILTRO APLICADO AQUI 🔴
    if OPERACAO_USUARIO != "Todas":
        admitidos = admitidos[admitidos["Operação"] == OPERACAO_USUARIO].copy()
    
    base_ativos = carregar_excel_primeira_aba(ARQ_ATIVOS)
    if OPERACAO_USUARIO != "Todas" and "Operação" in base_ativos.columns:
        base_ativos = base_ativos[base_ativos["Operação"] == OPERACAO_USUARIO].copy()

    nps = carregar_excel_primeira_aba(ARQ_NPS)
    nps = renomear_colunas_duplicadas(nps)

    batepapo = carregar_excel_primeira_aba(ARQ_BATEPAPO)
    batepapo = renomear_colunas_duplicadas(batepapo)

except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}")
    st.stop()

# =========================
# Pipeline principal
# =========================
try:
    base_oper = preparar_base_operacional(admitidos, base_ativos)
    base_oper = classificar_status_colaborador(base_oper, base_ativos)

    result = vincular_checks_otimizado(base_oper, nps, batepapo)
    df_nps = result["nps_vinculado"]
    df_bp = result["batepapo_vinculado"]
except Exception as e:
    st.error(f"Erro no pipeline de mentoria: {e}")
    st.stop()

# =========================
# O RESTO DO SEU CÓDIGO CONTINUA IGUAL...
# =========================
# (as funções filtrar_respostas_por_sidebar, render_farol, etc.
#  e a parte final dos tabs e gráficos permanecem IGUAIS ao seu código original)

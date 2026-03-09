import re
import unicodedata
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# Config da página
# =========================
st.set_page_config(
    page_title="Dashboard RH",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Imports opcionais p/ fuzzy
# =========================
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# =========================
# Título
# =========================
st.title("🚦 Farol de execução do Processo de Padrinhos")

# =========================
# Estilo
# =========================
st.markdown("""
<style>
.stApp { background: #0b0b0b; color: #f5f5f5; }
h1, h2, h3 { color: #f0d36b !important; }

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

/* Cards KPI das respostas */
.kpi-card {
    background: #171717;
    border: 1px solid #2d2d2d;
    border-radius: 16px;
    padding: 14px 14px 12px 14px;
    margin-bottom: 12px;
    min-height: 220px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

.kpi-title {
    color: #ffffff;
    font-weight: 700;
    font-size: 0.96rem;
    line-height: 1.3;
    min-height: 62px;
    margin-bottom: 14px;
}

.kpi-main {
    color: #ffffff;
    font-weight: 800;
    font-size: 2rem;
    line-height: 1;
    margin-bottom: 12px;
}

.kpi-sub {
    color: #cfcfcf;
    font-size: 0.88rem;
    margin-bottom: 10px;
}

.kpi-progress-wrap {
    width: 100%;
    height: 14px;
    background: #2b2b2b;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 12px;
    border: 1px solid #3a3a3a;
}

.kpi-progress-bar {
    height: 100%;
    border-radius: 999px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
}

.kpi-mini {
    background: #202020;
    border: 1px solid #303030;
    border-radius: 10px;
    padding: 8px 8px;
    text-align: center;
}

.kpi-mini-label {
    color: #bfbfbf;
    font-size: 0.78rem;
    margin-bottom: 4px;
}

.kpi-mini-value {
    color: #ffffff;
    font-size: 1rem;
    font-weight: 700;
}

.kpi-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def clean_cpf(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\D", "", s)
    if len(s) == 10:
        s = "0" + s
    return s if len(s) == 11 else ""

def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_text_nome_flex(x) -> str:
    s = norm_text(x)
    if not s:
        return ""
    tokens = [t for t in s.split() if len(t) > 1]
    return " ".join(tokens)

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

def similaridade_nome(resp_nome: str, cand_nome: str) -> float:
    if not resp_nome or not cand_nome:
        return 0

    resp_nome = norm_text_nome_flex(resp_nome)
    cand_nome = norm_text_nome_flex(cand_nome)

    if not resp_nome or not cand_nome:
        return 0

    if RAPIDFUZZ_OK:
        s1 = fuzz.token_set_ratio(resp_nome, cand_nome)
        s2 = fuzz.token_sort_ratio(resp_nome, cand_nome)
        s3 = fuzz.WRatio(resp_nome, cand_nome)
        s4 = fuzz.partial_ratio(resp_nome, cand_nome)
        seq = sequence_ratio(resp_nome, cand_nome)

        score = max(s1, s2 * 0.99, s3 * 0.98, s4 * 0.97, seq * 0.96)

        inter = token_overlap(resp_nome, cand_nome)
        if inter >= 2:
            score += 2.5
        if inter >= 3:
            score += 1.5
        if primeiro_nome(resp_nome) and primeiro_nome(resp_nome) == primeiro_nome(cand_nome):
            score += 2.5
        if ult_nome(resp_nome) and ult_nome(resp_nome) == ult_nome(cand_nome):
            score += 2.0

        return min(score, 100)

    base = max(
        sequence_ratio(resp_nome, cand_nome),
        sequence_ratio(" ".join(sorted(resp_nome.split())), " ".join(sorted(cand_nome.split())))
    )
    inter = token_overlap(resp_nome, cand_nome)
    if inter >= 2:
        base += 3
    if inter >= 3:
        base += 2
    if primeiro_nome(resp_nome) == primeiro_nome(cand_nome):
        base += 2
    if ult_nome(resp_nome) == ult_nome(cand_nome):
        base += 2
    return min(base, 100)

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

@st.cache_data(show_spinner=True)
def carregar_excel_primeira_aba(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    xls = pd.ExcelFile(path)
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

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
# Match de nomes
# =========================
def buscar_melhor_candidato_por_nome(nome_resp, base_lookup, op_resp="", score_min=82):
    if base_lookup.empty or not nome_resp:
        return None, 0

    pool = base_lookup.copy()

    nome_resp_norm = norm_text_nome_flex(nome_resp)
    op_resp_norm = norm_text(op_resp)

    if op_resp_norm and "op_norm" in pool.columns:
        pool_op = pool[pool["op_norm"] == op_resp_norm].copy()
        if not pool_op.empty:
            pool = pool_op

    pool = pool.dropna(subset=["nome_norm"]).copy()
    pool = pool[pool["nome_norm"].astype(str).str.strip() != ""].copy()

    if pool.empty:
        return None, 0

    pool["nome_norm_flex"] = pool["nome_norm"].apply(norm_text_nome_flex)

    if RAPIDFUZZ_OK:
        candidatos = process.extract(
            nome_resp_norm,
            pool["nome_norm_flex"].tolist(),
            scorer=fuzz.token_set_ratio,
            limit=15,
        )
        nomes_top = [c[0] for c in candidatos] if candidatos else []
        if nomes_top:
            pool = pool[pool["nome_norm_flex"].isin(nomes_top)].copy()

    pool["score_nome"] = pool["nome_norm_flex"].apply(lambda x: similaridade_nome(nome_resp_norm, x))
    pool["token_overlap"] = pool["nome_norm_flex"].apply(lambda x: token_overlap(nome_resp_norm, x))
    pool["primeiro_nome_ok"] = pool["nome_norm_flex"].apply(lambda x: primeiro_nome(nome_resp_norm) == primeiro_nome(x))
    pool["ultimo_nome_ok"] = pool["nome_norm_flex"].apply(lambda x: ult_nome(nome_resp_norm) == ult_nome(x))

    pool = pool.sort_values(
        ["score_nome", "token_overlap", "primeiro_nome_ok", "ultimo_nome_ok"],
        ascending=[False, False, False, False]
    )

    best = pool.head(1)
    if best.empty:
        return None, 0

    row = best.iloc[0]
    score = float(row["score_nome"])

    if score < score_min:
        return None, score

    if row["token_overlap"] == 0 and score < 90:
        return None, score

    return row, score

# =========================
# Base operacional
# =========================
def preparar_base_operacional(admitidos: pd.DataFrame, base_ativos: pd.DataFrame) -> pd.DataFrame:
    req_adm = ["Colaborador", "CPF", "Cargo", "Data", "Operação"]
    for c in req_adm:
        if c not in admitidos.columns:
            raise KeyError(f"Na planilha Admitidos não encontrei a coluna '{c}'")

    req_atv = ["Cargo", "Tipo Cargo"]
    for c in req_atv:
        if c not in base_ativos.columns:
            raise KeyError(f"Na Base colaboradores ativos não encontrei a coluna '{c}'")

    adm = admitidos.copy()
    atv = base_ativos.copy()

    adm["Data_dt"] = pd.to_datetime(adm["Data"], errors="coerce", dayfirst=True)
    adm["cpf_clean"] = adm["CPF"].apply(clean_cpf)
    adm["nome_norm"] = adm["Colaborador"].apply(norm_text)
    adm["cargo_norm"] = adm["Cargo"].apply(norm_text)
    adm["op_norm"] = adm["Operação"].apply(norm_text)

    atv["cargo_norm"] = atv["Cargo"].apply(norm_text)

    merged = adm.merge(
        atv[["cargo_norm", "Tipo Cargo"]].drop_duplicates(),
        on="cargo_norm",
        how="left",
    )

    merged["cargo_match_score"] = pd.NA

    if RAPIDFUZZ_OK:
        falt = merged["Tipo Cargo"].isna()
        if falt.any():
            cargos_ref = (
                atv[["cargo_norm", "Tipo Cargo"]]
                .dropna(subset=["cargo_norm", "Tipo Cargo"])
                .drop_duplicates()
            )
            ref_list = cargos_ref["cargo_norm"].tolist()
            ref_map = dict(zip(cargos_ref["cargo_norm"], cargos_ref["Tipo Cargo"]))

            def fuzzy_tipo(cargo_norm):
                if not cargo_norm:
                    return None, 0
                hit = process.extractOne(cargo_norm, ref_list, scorer=fuzz.WRatio)
                if not hit:
                    return None, 0
                best, score, _ = hit
                return ref_map.get(best), score

            tipo_score = merged.loc[falt, "cargo_norm"].apply(fuzzy_tipo)
            merged.loc[falt, "Tipo Cargo"] = tipo_score.apply(lambda t: t[0])
            merged.loc[falt, "cargo_match_score"] = tipo_score.apply(lambda t: t[1])

    merged["Tipo Cargo"] = merged["Tipo Cargo"].fillna("")
    oper = merged[merged["Tipo Cargo"].str.upper().eq("OPERACIONAL LOGÍSTICO")].copy()

    oper = oper[oper["Data_dt"] >= pd.Timestamp("2024-10-03")].copy()

    oper["Operação"] = oper["Operação"].astype(str).str.strip()
    oper["op_norm"] = oper["Operação"].apply(norm_text)

    corte_petropolis = pd.Timestamp("2025-08-01")
    corte_vidros_pr = pd.Timestamp("2026-02-01")
    corte_ponta_grossa = pd.Timestamp("2025-09-01")

    mask_petropolis = oper["op_norm"].eq("CD PETROPOLIS")
    mask_vidros_pr = oper["op_norm"].eq("VIDROS PR")
    mask_ponta_grossa = oper["op_norm"].str.contains("PONTA GROSSA", na=False)

    oper = oper[
        (~mask_petropolis | (oper["Data_dt"] >= corte_petropolis)) &
        (~mask_vidros_pr | (oper["Data_dt"] >= corte_vidros_pr)) &
        (~mask_ponta_grossa | (oper["Data_dt"] >= corte_ponta_grossa))
    ].copy()

    return oper

# =========================
# Status colaborador
# =========================
def classificar_status_colaborador(base_oper: pd.DataFrame, base_ativos: pd.DataFrame) -> pd.DataFrame:
    base = base_oper.copy()
    atv = base_ativos.copy()

    if "CPF" in atv.columns:
        atv["cpf_clean"] = atv["CPF"].apply(clean_cpf)
    else:
        atv["cpf_clean"] = ""

    if "Colaborador" in atv.columns:
        atv["nome_norm"] = atv["Colaborador"].apply(norm_text)
    else:
        atv["nome_norm"] = ""

    if "Operação" in atv.columns:
        atv["op_norm"] = atv["Operação"].apply(norm_text)
    else:
        atv["op_norm"] = ""

    base["Status Colaborador"] = "Inativo"
    base["match_ativo_tipo"] = "NAO_ENCONTRADO"
    base["match_ativo_score"] = pd.NA

    ativos_cpf = set(atv.loc[atv["cpf_clean"].astype(str).str.len() == 11, "cpf_clean"].unique().tolist())
    mask_cpf = base["cpf_clean"].isin(ativos_cpf)
    base.loc[mask_cpf, "Status Colaborador"] = "Ativo"
    base.loc[mask_cpf, "match_ativo_tipo"] = "CPF"

    falt = base["Status Colaborador"].eq("Inativo")
    ativos_nome_exato = set(atv.loc[atv["nome_norm"].astype(str).str.strip() != "", "nome_norm"].unique().tolist())
    mask_nome = falt & base["nome_norm"].isin(ativos_nome_exato)
    base.loc[mask_nome, "Status Colaborador"] = "Ativo"
    base.loc[mask_nome, "match_ativo_tipo"] = "NOME_EXATO"

    falt = base["Status Colaborador"].eq("Inativo")
    if falt.any():
        atv_lookup = atv[["nome_norm", "op_norm"]].drop_duplicates().copy()
        atv_lookup = atv_lookup[atv_lookup["nome_norm"].astype(str).str.strip() != ""].copy()

        def match_ativo(row):
            nome = row.get("nome_norm", "")
            op = row.get("op_norm", "")
            _, score = buscar_melhor_candidato_por_nome(
                nome_resp=nome,
                base_lookup=atv_lookup,
                op_resp=op,
                score_min=88
            )
            return score

        if not atv_lookup.empty:
            base.loc[falt, "match_ativo_score"] = base.loc[falt].apply(match_ativo, axis=1)
            idx_ok = base.loc[falt].index[base.loc[falt, "match_ativo_score"].fillna(0) >= 88]
            base.loc[idx_ok, "Status Colaborador"] = "Ativo"
            base.loc[idx_ok, "match_ativo_tipo"] = "NOME_FUZZY"

    return base

# =========================
# Vincular respostas
# =========================
def vincular_checks(base_oper: pd.DataFrame, nps: pd.DataFrame, batepapo: pd.DataFrame) -> dict:
    base = base_oper.copy()
    base["cpf_clean"] = base["cpf_clean"].fillna("")
    base["nome_norm"] = base["nome_norm"].fillna("")
    base["op_norm"] = base["Operação"].apply(norm_text)

    nps_df = nps.copy()
    nps_nome_col = "Informe seu nome completo:"
    nps_cpf_col = "Informe seu CPF:"
    nps_op_col = "Informe a operação que você trabalha:"

    for c in [nps_nome_col, nps_cpf_col, "Data Cadastro"]:
        if c not in nps_df.columns:
            raise KeyError(f"No NPS Mentor não encontrei a coluna '{c}'")

    nps_df["cpf_clean"] = nps_df[nps_cpf_col].apply(clean_cpf)
    nps_df["nome_norm"] = nps_df[nps_nome_col].apply(norm_text)
    nps_df["op_norm"] = nps_df[nps_op_col].apply(norm_text) if nps_op_col in nps_df.columns else ""
    nps_df["Data Cadastro"] = pd.to_datetime(nps_df["Data Cadastro"], errors="coerce", dayfirst=True)
    nps_df["DataHora Resposta"] = combinar_data_hora(nps_df, "Data Cadastro", "Horário da resposta")

    bp = batepapo.copy()
    bp_nome_col = "Insira o nome do colaborador:"
    bp_cpf_col = "Inserir o CPF do colaborador:"

    for c in [bp_nome_col, bp_cpf_col, "Data Cadastro"]:
        if c not in bp.columns:
            raise KeyError(f"No Bate papo mentor não encontrei a coluna '{c}'")

    bp["cpf_clean"] = bp[bp_cpf_col].apply(clean_cpf)
    bp["nome_norm"] = bp[bp_nome_col].apply(norm_text)
    bp["op_norm"] = ""
    bp["Data Cadastro"] = pd.to_datetime(bp["Data Cadastro"], errors="coerce", dayfirst=True)
    bp["DataHora Resposta"] = combinar_data_hora(bp, "Data Cadastro", "Horário da resposta")

    base_cols = [
        "cpf_clean", "Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação",
        "Data", "Data_dt", "nome_norm", "op_norm", "Status Colaborador"
    ]

    nps_m = nps_df.merge(base[base_cols], on="cpf_clean", how="left", suffixes=("", "_base"))
    bp_m = bp.merge(base[base_cols], on="cpf_clean", how="left", suffixes=("", "_base"))

    nps_m["match_tipo"] = nps_m["Colaborador"].apply(lambda x: "CPF" if pd.notna(x) else "NAO_ENCONTRADO")
    bp_m["match_tipo"] = bp_m["Colaborador"].apply(lambda x: "CPF" if pd.notna(x) else "NAO_ENCONTRADO")
    nps_m["match_score"] = pd.NA
    bp_m["match_score"] = pd.NA

    base_lookup = base.copy()

    def fuzzy_match_row(row):
        nome = row.get("nome_norm", "")
        op = row.get("op_norm", "")
        if not nome:
            return None, 0

        cand, score = buscar_melhor_candidato_por_nome(
            nome_resp=nome,
            base_lookup=base_lookup,
            op_resp=op,
            score_min=82
        )
        return cand, score

    falt = nps_m["Colaborador"].isna()
    if falt.any():
        resultados = nps_m.loc[falt].apply(lambda r: fuzzy_match_row(r), axis=1)
        resultados_dict = resultados.to_dict()

        nps_m.loc[falt, "match_score"] = pd.Series({idx: val[1] for idx, val in resultados_dict.items()})
        idx_ok = [idx for idx, val in resultados_dict.items() if val[0] is not None and val[1] >= 82]

        for idx in idx_ok:
            base_row, _ = resultados_dict[idx]
            for col in ["Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação", "Data", "Data_dt", "Status Colaborador"]:
                nps_m.at[idx, col] = base_row[col]
            nps_m.at[idx, "match_tipo"] = "NOME_FUZZY"

    falt = bp_m["Colaborador"].isna()
    if falt.any():
        resultados = bp_m.loc[falt].apply(lambda r: fuzzy_match_row(r), axis=1)
        resultados_dict = resultados.to_dict()

        bp_m.loc[falt, "match_score"] = pd.Series({idx: val[1] for idx, val in resultados_dict.items()})
        idx_ok = [idx for idx, val in resultados_dict.items() if val[0] is not None and val[1] >= 82]

        for idx in idx_ok:
            base_row, _ = resultados_dict[idx]
            for col in ["Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação", "Data", "Data_dt", "Status Colaborador"]:
                bp_m.at[idx, col] = base_row[col]
            bp_m.at[idx, "match_tipo"] = "NOME_FUZZY"

    return {
        "base_operacional": base,
        "nps_vinculado": nps_m,
        "batepapo_vinculado": bp_m,
    }

# =========================
# Etapas
# =========================
ETAPAS = [
    {
        "chave": "NPS_1_SEMANA",
        "titulo": "NPS 1ª SEMANA",
        "tipo": "NPS",
        "campo_selecao": "Selecione a semana da avaliação:",
        "valor_selecao": "Primeira semana junto ao padrinho.",
        "prazo_min_dias": 11,
        "prazo_max_dias": 14,
    },
    {
        "chave": "NPS_ULTIMA",
        "titulo": "NPS ÚLTIMA SEMANA",
        "tipo": "NPS",
        "campo_selecao": "Selecione a semana da avaliação:",
        "valor_selecao": "Última semana junto ao padrinho.",
        "prazo_min_dias": 20,
        "prazo_max_dias": 32,
    },
    {
        "chave": "BP_2_SEMANA",
        "titulo": "BATE-PAPO — 2ª SEMANA",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Segunda Semana",
        "prazo_min_dias": 11,
        "prazo_max_dias": 14,
    },
    {
        "chave": "BP_3_SEMANA",
        "titulo": "BATE-PAPO — 3ª SEMANA",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Terceira Semana",
        "prazo_min_dias": 20,
        "prazo_max_dias": 22,
    },
    {
        "chave": "BP_ULTIMA",
        "titulo": "BATE-PAPO — ÚLTIMA SEMANA",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Última Semana",
        "prazo_min_dias": 28,
        "prazo_max_dias": 32,
    },
]

def status_prazo(data_realizacao, prazo_min, prazo_max, hoje):
    if pd.isna(data_realizacao):
        return "Não realizado - Atenção" if hoje <= prazo_max else "Não realizado - Fora do prazo"
    if data_realizacao < prazo_min:
        return "Realizado antes do prazo"
    if data_realizacao <= prazo_max:
        return "Realizado no prazo"
    return "Realizado fora do prazo"

def dias_para_prazo_max(prazo_max, hoje):
    if pd.isna(prazo_max):
        return pd.NA
    return (prazo_max.normalize() - hoje.normalize()).days

def montar_farol_por_etapa(base_oper, df_nps, df_bp, hoje):
    base = base_oper.copy()
    if "Data_dt" not in base.columns:
        base["Data_dt"] = pd.to_datetime(base["Data"], errors="coerce", dayfirst=True)

    farois = {}
    for etapa in ETAPAS:
        tmp = base[
            ["Colaborador", "CPF", "cpf_clean", "Operação", "Cargo", "Data_dt", "Status Colaborador"]
        ].copy()

        tmp["Prazo Mín"] = tmp["Data_dt"] + pd.to_timedelta(etapa["prazo_min_dias"], unit="D")
        tmp["Prazo Máx"] = tmp["Data_dt"] + pd.to_timedelta(etapa["prazo_max_dias"], unit="D")

        form = df_nps if etapa["tipo"] == "NPS" else df_bp
        campo = etapa["campo_selecao"]
        valor = etapa["valor_selecao"]

        if campo not in form.columns:
            tmp["Data Realização"] = pd.NaT
        else:
            form_et = form[form[campo].astype(str).str.strip().eq(valor)].copy()
            col_realizacao = "DataHora Resposta" if "DataHora Resposta" in form_et.columns else "Data Cadastro"

            real = (
                form_et.dropna(subset=[col_realizacao])
                .dropna(subset=["Colaborador"])
                .groupby("Colaborador", as_index=False)[col_realizacao]
                .min()
                .rename(columns={col_realizacao: "Data Realização"})
            )

            tmp = tmp.merge(real, on="Colaborador", how="left")

        tmp["Status"] = tmp.apply(
            lambda r: status_prazo(r["Data Realização"], r["Prazo Mín"], r["Prazo Máx"], hoje),
            axis=1,
        )
        tmp["Dias p/ Prazo Máx"] = tmp["Prazo Máx"].apply(lambda d: dias_para_prazo_max(d, hoje))

        ordem = pd.CategoricalDtype(
            categories=[
                "Não realizado - Fora do prazo",
                "Não realizado - Atenção",
                "Realizado fora do prazo",
                "Realizado no prazo",
                "Realizado antes do prazo",
            ],
            ordered=True,
        )
        tmp["Status"] = tmp["Status"].astype(ordem)
        tmp = tmp.sort_values(["Status", "Dias p/ Prazo Máx"], ascending=[True, True])

        farois[etapa["chave"]] = tmp

    return farois

def render_farol(df_farol: pd.DataFrame, titulo: str, key_prefix: str):
    if df_farol.empty:
        st.info("Sem dados para os filtros selecionados.")
        return

    df_farol = df_farol.copy()
    df_farol["Operação"] = df_farol["Operação"].fillna("SEM OPERAÇÃO").astype(str).str.strip()
    df_farol.loc[df_farol["Operação"].eq(""), "Operação"] = "SEM OPERAÇÃO"

    st.markdown(
        f'<div class="card"><h3 style="margin:0; text-align:center;">{titulo}</h3></div>',
        unsafe_allow_html=True
    )

    total = len(df_farol)
    pend_fora = int((df_farol["Status"] == "Não realizado - Fora do prazo").sum())

    mask_atenc_7 = (
        (df_farol["Status"] == "Não realizado - Atenção") &
        (pd.to_numeric(df_farol["Dias p/ Prazo Máx"], errors="coerce") <= 7)
    )
    pend_atenc_7 = int(mask_atenc_7.sum())

    realizados = int(
        df_farol["Status"].isin(["Realizado no prazo", "Realizado fora do prazo", "Realizado antes do prazo"]).sum()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{total:,}".replace(",", "."))
    c2.metric("🔴 Pendentes em atraso", f"{pend_fora:,}".replace(",", "."))
    c3.metric("🟡 No prazo vencendo em até 7 dias", f"{pend_atenc_7:,}".replace(",", "."))
    c4.metric("Realizados", f"{realizados:,}".replace(",", "."))

    st.markdown("<hr/>", unsafe_allow_html=True)

    g = (
        df_farol.assign(pend_fora=(df_farol["Status"] == "Não realizado - Fora do prazo"))
        .groupby("Operação", as_index=False)
        .agg(total=("Colaborador", "count"), pend_fora=("pend_fora", "sum"))
    )
    g["Aderência %"] = ((g["total"] - g["pend_fora"]) / g["total"]).fillna(0) * 100
    g["Faixa"] = g["Aderência %"].apply(classificar_faixa_aderencia)
    g = g.sort_values("Aderência %", ascending=False)

    fig = px.bar(
        g,
        x="Operação",
        y="Aderência %",
        text=g["Aderência %"].round(2).astype(str) + "%",
        color="Faixa",
        color_discrete_map={
            ">= 90%": "#2e7d32",
            "80% a 89%": "#f0d36b",
            "< 80%": "#c62828",
        },
        category_orders={"Faixa": [">= 90%", "80% a 89%", "< 80%"]},
    )
    fig.update_traces(textposition="outside", cliponaxis=False, width=0.48)
    fig.update_layout(
        height=380,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis=dict(range=[0, 100], title="", showgrid=False, zeroline=False, showticklabels=False, visible=False),
        xaxis=dict(title="", showgrid=False),
        legend_title="Faixa",
        paper_bgcolor="#0b0b0b",
        plot_bgcolor="#1a1a1a",
        bargap=0.45,
    )
    st.plotly_chart(fig, width="stretch", key=f"chart_{key_prefix}")

    tabela = df_farol.copy()
    cols_show = [
        "Status Colaborador", "Operação", "Colaborador", "CPF", "Cargo",
        "Data_dt", "Prazo Mín", "Prazo Máx", "Data Realização",
        "Dias p/ Prazo Máx", "Status"
    ]
    cols_show = [c for c in cols_show if c in tabela.columns]
    tabela = tabela[cols_show].rename(columns={"Data_dt": "Data Admissão"})
    tabela = formatar_datas_para_tabela(tabela)

    st.dataframe(
        style_table(tabela),
        width="stretch",
        height=430,
        key=f"df_{key_prefix}"
    )

    excel_bytes = to_excel_bytes(tabela, sheet_name="farol")
    st.download_button(
        "⬇️ Baixar Excel",
        data=excel_bytes,
        file_name=f"farol_padrinhos_{key_prefix.lower()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{key_prefix}",
        width="stretch"
    )

# =========================
# Respostas
# =========================
def filtrar_respostas_por_sidebar(df_resp: pd.DataFrame) -> pd.DataFrame:
    df = df_resp.copy()

    if "Operação" in df.columns and filtro_ops:
        df = df[df["Operação"].isin(filtro_ops)]

    if "Data_dt" in df.columns:
        df = df[(df["Data_dt"] >= dt_ini) & (df["Data_dt"] <= dt_fim)]

    if "Cargo" in df.columns and filtro_cargos:
        df = df[df["Cargo"].isin(filtro_cargos)]

    if "Status Colaborador" in df.columns and filtro_status_colaborador:
        df = df[df["Status Colaborador"].isin(filtro_status_colaborador)]

    if "Colaborador" in df.columns and filtro_colaborador:
        df = df[df["Colaborador"].isin(filtro_colaborador)]

    return df

def render_grafico_resposta(df: pd.DataFrame, coluna: str, key_prefix: str):
    serie = (
        df[coluna]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .dropna()
    )

    if serie.empty:
        st.info(f"Sem dados para: {coluna}")
        return

    base = (
        serie.value_counts(dropna=False)
        .rename_axis("Resposta")
        .reset_index(name="Qtd")
    )

    def normalizar_resposta(v):
        vv = norm_text(v)
        if vv == "SIM":
            return "Sim"
        if vv in {"NAO", "NÃO"}:
            return "Não"
        return str(v).strip()

    base["Resposta"] = base["Resposta"].apply(normalizar_resposta)
    base = base.groupby("Resposta", as_index=False)["Qtd"].sum()

    total = int(base["Qtd"].sum())
    qtd_sim = int(base.loc[base["Resposta"].eq("Sim"), "Qtd"].sum())
    qtd_nao = int(base.loc[base["Resposta"].eq("Não"), "Qtd"].sum())
    outros = max(total - qtd_sim - qtd_nao, 0)

    perc_sim = (qtd_sim / total * 100) if total > 0 else 0

    if perc_sim >= 90:
        cor_barra = "#79c257"
        cor_badge_bg = "#1f3a22"
        cor_badge_txt = "#a7e08f"
        status_txt = "Aderência alta"
    elif perc_sim >= 80:
        cor_barra = "#f0d36b"
        cor_badge_bg = "#3d3517"
        cor_badge_txt = "#f6df8b"
        status_txt = "Aderência moderada"
    else:
        cor_barra = "#d9534f"
        cor_badge_bg = "#3d1d1b"
        cor_badge_txt = "#f2a29f"
        status_txt = "Ponto de atenção"

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="color:#ffffff; font-weight:700; font-size:0.96rem; line-height:1.35; min-height:52px; margin-bottom:10px;">
                {coluna}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="color:#ffffff; font-weight:800; font-size:2.1rem; line-height:1; margin-bottom:6px;">
                {perc_sim:.1f}%
            </div>
            <div style="color:#cfcfcf; font-size:0.88rem; margin-bottom:12px;">
                Percentual de respostas positivas
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="width:100%; height:14px; background:#2b2b2b; border-radius:999px; overflow:hidden; border:1px solid #3a3a3a; margin-bottom:14px;">
                <div style="width:{perc_sim:.1f}%; height:100%; background:{cor_barra}; border-radius:999px;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Sim", qtd_sim)
        c2.metric("Não", qtd_nao)
        c3.metric("Total", total)

        if outros > 0:
            st.markdown(
                f"""
                <div style="margin-top:8px; color:#cfcfcf; font-size:0.84rem;">
                    Outras respostas: <b style="color:#ffffff;">{outros}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div style="margin-top:10px; display:inline-block; background:{cor_badge_bg}; color:{cor_badge_txt};
                        padding:6px 10px; border-radius:999px; font-size:0.78rem; font-weight:700;">
                {status_txt}
            </div>
            """,
            unsafe_allow_html=True
        )

def render_lista_respostas(df: pd.DataFrame, coluna: str):
    titulo_limpo = str(coluna).split("__")[0].split(".")[0]

    textos = (
        df[coluna]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-": pd.NA})
        .dropna()
        .tolist()
    )

    st.markdown(
        f"""
        <div class="lista-box">
            <div class="lista-titulo">{titulo_limpo}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container(height=520, border=False):
        if not textos:
            st.write("Sem respostas.")
            return

        for txt in textos:
            st.markdown(f"""
<div style="padding:6px 4px;border-bottom:1px solid #2a2a2a;">
{txt}
</div>
""", unsafe_allow_html=True)
    )

def render_info_colaborador(df: pd.DataFrame, coluna_padrinho: str):
    if not filtro_colaborador or len(filtro_colaborador) != 1:
        return

    if df.empty:
        return

    base = df.copy()
    nome_sel = filtro_colaborador[0]
    base = base[base["Colaborador"].astype(str).eq(nome_sel)].copy()

    if base.empty:
        return

    colab = base["Colaborador"].dropna().astype(str).iloc[0] if "Colaborador" in base.columns and not base["Colaborador"].dropna().empty else "-"
    cargo = base["Cargo"].dropna().astype(str).iloc[0] if "Cargo" in base.columns and not base["Cargo"].dropna().empty else "-"
    padrinho = base[coluna_padrinho].dropna().astype(str).iloc[0] if coluna_padrinho in base.columns and not base[coluna_padrinho].dropna().empty else "-"

    st.markdown(
        f'<div class="info-box"><b>Colaborador:</b> {colab} &nbsp;&nbsp;&nbsp; '
        f'<b>Cargo:</b> {cargo} &nbsp;&nbsp;&nbsp; '
        f'<b>Padrinho:</b> {padrinho}</div>',
        unsafe_allow_html=True
    )

# =========================
# Paths
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARQ_ADMITIDOS = DATA_DIR / "Admitidos.xlsx"
ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"
ARQ_NPS = DATA_DIR / "NPS Mentor.xlsx"
ARQ_BATEPAPO = DATA_DIR / "Bate papo mentor.xlsx"

# =========================
# Carregamento
# =========================
try:
    admitidos = carregar_excel_primeira_aba(ARQ_ADMITIDOS)
    base_ativos = carregar_excel_primeira_aba(ARQ_ATIVOS)

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

    result = vincular_checks(base_oper, nps, batepapo)
    df_nps = result["nps_vinculado"]
    df_bp = result["batepapo_vinculado"]
except Exception as e:
    st.error(f"Erro no pipeline de mentoria: {e}")
    st.stop()

# =========================
# Cards topo
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Admitidos (arquivo)", f"{len(admitidos):,}".replace(",", "."))
c2.metric("Operacional Logístico (base do farol)", f"{len(base_oper):,}".replace(",", "."))
c3.metric("NPS (linhas)", f"{len(df_nps):,}".replace(",", "."))
c4.metric("Bate-papo (linhas)", f"{len(df_bp):,}".replace(",", "."))

# =========================
# Sidebar — filtros
# =========================
st.sidebar.markdown("## 🔎 Filtros")

adm_datas = admitidos.copy()
adm_datas["Data_dt"] = pd.to_datetime(adm_datas["Data"], errors="coerce", dayfirst=True)

data_min_disponivel = adm_datas["Data_dt"].min()
data_max_disponivel = adm_datas["Data_dt"].max()

if pd.isna(data_min_disponivel):
    data_min_disponivel = pd.Timestamp("2024-10-03")
if pd.isna(data_max_disponivel):
    data_max_disponivel = pd.Timestamp(datetime.now().date())

data_padrao_ini = pd.Timestamp("2025-01-01")
if data_padrao_ini < data_min_disponivel:
    data_padrao_ini = data_min_disponivel

ops_all = sorted([x for x in base_oper["Operação"].fillna("").astype(str).unique().tolist() if x.strip()])
cargos_all = sorted([x for x in base_oper["Cargo"].fillna("").astype(str).unique().tolist() if x.strip()])
colab_all = sorted([x for x in base_oper["Colaborador"].fillna("").astype(str).unique().tolist() if x.strip()])

status_options = [
    "Não realizado - Fora do prazo",
    "Não realizado - Atenção",
    "Realizado fora do prazo",
    "Realizado no prazo",
    "Realizado antes do prazo",
]

status_colaborador_options = ["Ativo", "Inativo"]

filtro_ops = st.sidebar.multiselect(
    "Operação",
    options=ops_all,
    default=[],
    key="f_ops_sidebar"
)

st.sidebar.markdown("### Período de admissão")
col_ini, col_fim = st.sidebar.columns(2)

with col_ini:
    dt_ini = st.date_input(
        "Início",
        value=data_padrao_ini.date(),
        min_value=data_min_disponivel.date(),
        max_value=data_max_disponivel.date(),
        format="DD/MM/YYYY",
        key="f_dt_ini_sidebar"
    )

with col_fim:
    dt_fim = st.date_input(
        "Fim",
        value=data_max_disponivel.date(),
        min_value=data_min_disponivel.date(),
        max_value=data_max_disponivel.date(),
        format="DD/MM/YYYY",
        key="f_dt_fim_sidebar"
    )

dt_ini = pd.Timestamp(dt_ini)
dt_fim = pd.Timestamp(dt_fim)

filtro_cargos = st.sidebar.multiselect(
    "Cargo",
    options=cargos_all,
    default=[],
    key="f_cargo_sidebar"
)

filtro_colaborador = st.sidebar.multiselect(
    "Colaborador",
    options=colab_all,
    default=[],
    key="f_colab_sidebar"
)

filtro_status_colaborador = st.sidebar.multiselect(
    "Status do colaborador",
    options=status_colaborador_options,
    default=["Ativo"],
    key="f_status_colab_sidebar"
)

filtro_status = st.sidebar.multiselect(
    "Status da etapa",
    options=status_options,
    default=[],
    key="f_status_etapa_sidebar"
)

def aplicar_filtros_farol(df_farol: pd.DataFrame) -> pd.DataFrame:
    df = df_farol.copy()

    if filtro_ops:
        df = df[df["Operação"].isin(filtro_ops)]

    if "Data_dt" in df.columns:
        df = df[(df["Data_dt"] >= dt_ini) & (df["Data_dt"] <= dt_fim)]

    if filtro_cargos:
        df = df[df["Cargo"].isin(filtro_cargos)]

    if filtro_colaborador:
        df = df[df["Colaborador"].isin(filtro_colaborador)]

    if filtro_status_colaborador:
        df = df[df["Status Colaborador"].isin(filtro_status_colaborador)]

    if filtro_status:
        df = df[df["Status"].isin(filtro_status)]

    return df

# =========================
# FAROL
# =========================
st.header("🚦 ADERÊNCIA — PROCESSO PADRINHOS")

hoje = pd.Timestamp(datetime.now().date())
farois = montar_farol_por_etapa(base_oper, df_nps, df_bp, hoje=hoje)

tabs = st.tabs([
    "PROCESSO PADRINHOS (GERAL)",
    "NPS 1ª SEMANA",
    "NPS ÚLTIMA SEMANA",
    "BATE-PAPO 2ª SEMANA",
    "BATE-PAPO 3ª SEMANA",
    "BATE-PAPO ÚLTIMA SEMANA",
])

with tabs[0]:
    df_all = pd.concat([farois[e["chave"]] for e in ETAPAS], ignore_index=True)
    df_all = aplicar_filtros_farol(df_all)
    render_farol(df_all, "PROCESSO PADRINHOS — ADERÊNCIA GERAL", key_prefix="GERAL")

with tabs[1]:
    df = aplicar_filtros_farol(farois["NPS_1_SEMANA"])
    render_farol(df, "NPS 1ª SEMANA", key_prefix="NPS1")

with tabs[2]:
    df = aplicar_filtros_farol(farois["NPS_ULTIMA"])
    render_farol(df, "NPS ÚLTIMA SEMANA", key_prefix="NPSU")

with tabs[3]:
    df = aplicar_filtros_farol(farois["BP_2_SEMANA"])
    render_farol(df, "BATE-PAPO — 2ª SEMANA", key_prefix="BP2")

with tabs[4]:
    df = aplicar_filtros_farol(farois["BP_3_SEMANA"])
    render_farol(df, "BATE-PAPO — 3ª SEMANA", key_prefix="BP3")

with tabs[5]:
    df = aplicar_filtros_farol(farois["BP_ULTIMA"])
    render_farol(df, "BATE-PAPO — ÚLTIMA SEMANA", key_prefix="BPU")

# =========================
# Acompanhamento processo padrinhos
# =========================
st.markdown("<br>", unsafe_allow_html=True)
st.header("📋 Acompanhamento Processo Padrinhos")

resp_tabs = st.tabs(["NPS Mentor", "Bate papo mentor"])

with resp_tabs[0]:
    nps_f = filtrar_respostas_por_sidebar(df_nps)
    render_info_colaborador(nps_f, "Informe o nome do seu Padrinho.")

    coluna_semana_nps = "Selecione a semana da avaliação:"

    colunas_primeira_semana = [
        "O seu padrinho te apresentou aos colegas, facilitando sua adaptação, e te fez conhecer melhor a empresa?",
        "O seu padrinho te ajudou a entender melhor a função que você irá exercer?",
        "O seu padrinho se mostrou presente e disposto a te ajudar?",
        "O seu padrinho demonstrou um bom conhecimento e habilidades no campo de trabalho?",
    ]

    colunas_ultima_semana = [
        "O seu padrinho fez e registrou o bate papo semanal com você na segunda, terceira e quarta semana de integração?",
        "O seu padrinho fez bate papo final com você na última semana de integração, informando seu desempenho no período?",
        "Você se sentiu confortável em discutir desafios e questões com seu padrinho?",
    ]

    colunas_texto_ultima = [
        "No que você acha que seu padrinho mandou bem?",
        "No que o seu padrinho precisa melhorar?",
        "Quais são suas sugestões para melhorar o programa de padrinhos?",
    ]

    if coluna_semana_nps in nps_f.columns:
        nps_primeira = nps_f[nps_f[coluna_semana_nps].astype(str).str.strip().eq("Primeira semana junto ao padrinho.")].copy()
        nps_ultima = nps_f[nps_f[coluna_semana_nps].astype(str).str.strip().eq("Última semana junto ao padrinho.")].copy()
    else:
        nps_primeira = pd.DataFrame()
        nps_ultima = pd.DataFrame()

    st.markdown('<div class="titulo-amarelo">➡️Primeira semana junto ao padrinho.</div>', unsafe_allow_html=True)

    if nps_primeira.empty:
        st.info("Sem respostas para os filtros selecionados.")
    else:
        cols = st.columns(2, gap="large")
        for i, coluna in enumerate(colunas_primeira_semana):
            if coluna in nps_primeira.columns:
                with cols[i % 2]:
                    render_grafico_resposta(nps_primeira, coluna, key_prefix=f"nps_p1_{i}")

        col_card = st.columns([1, 3])[0]
        with col_card:
            render_card_data_resposta(nps_primeira, titulo="Registro da resposta")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="titulo-amarelo">➡️Última semana junto ao padrinho.</div>', unsafe_allow_html=True)

    if nps_ultima.empty:
        st.info("Sem respostas para os filtros selecionados.")
    else:
        cols = st.columns(2, gap="large")
        for i, coluna in enumerate(colunas_ultima_semana):
            if coluna in nps_ultima.columns:
                with cols[i % 2]:
                    render_grafico_resposta(nps_ultima, coluna, key_prefix=f"nps_p2_{i}")

        col_card2 = st.columns([1, 3])[0]
        with col_card2:
            render_card_data_resposta(nps_ultima, titulo="Registro da resposta")

        textos_existentes = [c for c in colunas_texto_ultima if c in nps_ultima.columns]
        if textos_existentes:
            st.markdown("<br>", unsafe_allow_html=True)
            cols_texto = st.columns(len(textos_existentes))
            for i, coltxt in enumerate(textos_existentes):
                with cols_texto[i]:
                    render_lista_respostas(nps_ultima, coltxt)

with resp_tabs[1]:
    bp_f = filtrar_respostas_por_sidebar(df_bp)
    render_info_colaborador(bp_f, "Informe o nome do padrinho:")

    coluna_semana_bp = "Selecione a semana do bate papo:"

    if coluna_semana_bp in bp_f.columns:
        bp_segunda = bp_f[bp_f[coluna_semana_bp].astype(str).str.strip().eq("Segunda Semana")].copy()
        bp_terceira = bp_f[bp_f[coluna_semana_bp].astype(str).str.strip().eq("Terceira Semana")].copy()
        bp_ultima = bp_f[bp_f[coluna_semana_bp].astype(str).str.strip().eq("Última Semana")].copy()
    else:
        bp_segunda = pd.DataFrame()
        bp_terceira = pd.DataFrame()
        bp_ultima = pd.DataFrame()

    st.markdown('<div class="titulo-amarelo">➡️Segunda Semana</div>', unsafe_allow_html=True)

    graficos_segunda = [
        "Você conhece quais são os EPIs obrigatórios da sua função?",
        "Você já se sentiu confortável em interagir com sua equipe de trabalho?",
        "De forma prática, você conseguiu executar suas atividades nessa primeira semana?",
        "Você sabe como utilizar as ferramentas de trabalho que fazem parte de sua rotina?",
    ]

    listas_segunda = [
        "Você tem alguma dúvida ou algo que eu possa te ajudar?",
        "Como foi sua rotina nesta semana?",
    ]

    if bp_segunda.empty:
        st.info("Sem respostas para os filtros selecionados.")
    else:
        cols = st.columns(2, gap="large")
        for i, coluna in enumerate(graficos_segunda):
            if coluna in bp_segunda.columns:
                with cols[i % 2]:
                    render_grafico_resposta(bp_segunda, coluna, key_prefix=f"bp_s2_{i}")

        col_card_bp2 = st.columns([1, 3])[0]
        with col_card_bp2:
            render_card_data_resposta(bp_segunda, titulo="Registro da resposta")

        cols_lista_seg = st.columns(len([c for c in listas_segunda if c in bp_segunda.columns]) or 1)
        idx = 0
        for coluna in listas_segunda:
            if coluna in bp_segunda.columns:
                with cols_lista_seg[idx]:
                    render_lista_respostas(bp_segunda, coluna)
                idx += 1

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="titulo-amarelo">➡️Terceira Semana</div>', unsafe_allow_html=True)

    graficos_terceira = [
        "Você conhece a rotina de reuniões semanais que você deve participar?",
        "Você sabe fazer relatos de segurança e abordagens positivas?",
        "Você entende o que é DPO / VPO?",
    ]

    listas_terceira = []
    for c in bp_terceira.columns:
        if "Como foi sua rotina nesta semana?" in c:
            listas_terceira.append(c)
        if "Você tem alguma dúvida ou algo que eu possa te ajudar?" in c:
            listas_terceira.append(c)

    if bp_terceira.empty:
        st.info("Sem respostas para os filtros selecionados.")
    else:
        cols = st.columns(2, gap="large")
        for i, coluna in enumerate(graficos_terceira):
            if coluna in bp_terceira.columns:
                with cols[i % 2]:
                    render_grafico_resposta(bp_terceira, coluna, key_prefix=f"bp_s3_{i}")

        col_card_bp3 = st.columns([1, 3])[0]
        with col_card_bp3:
            render_card_data_resposta(bp_terceira, titulo="Registro da resposta")

        listas_terceira_exist = [c for c in listas_terceira if c in bp_terceira.columns]
        if listas_terceira_exist:
            cols_lista_ter = st.columns(len(listas_terceira_exist))
            for idx, coluna in enumerate(listas_terceira_exist):
                with cols_lista_ter[idx]:
                    render_lista_respostas(bp_terceira, coluna)
        else:
            st.info("Sem respostas abertas para a Terceira Semana.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="titulo-amarelo">➡️Última Semana</div>', unsafe_allow_html=True)

    graficos_ultima = [
        "Você se sente integrado a empresa?",
        "Você conhece o código de conduta da empresa?",
        "O novo(a) colaborador(a) se demonstrou comprometido com as atividades e disposto a aprender?",
        "O novo(a) colaborador(a) demonstrou progresso satisfatório desde o início da integração?",
    ]

    listas_ultima = [
        "Quais são as principais dúvidas ou dificuldades que você ainda enfrenta em sua função?",
        "Comente sobre sua adaptação à unidade, função e empresa",
        "O que você mais gosta hoje na empresa, após toda a integração? E o que você acha que pode melhorar?",
        "Descreva no que você acha que o novo colaborador mandou bem.",
        "Descreva os pontos que o novo colaborador ainda necessita de acompanhamento e que podem contribuir na avaliação de desempenho.",
    ]

    if bp_ultima.empty:
        st.info("Sem respostas para os filtros selecionados.")
    else:
        cols = st.columns(2, gap="large")
        for i, coluna in enumerate(graficos_ultima):
            if coluna in bp_ultima.columns:
                with cols[i % 2]:
                    render_grafico_resposta(bp_ultima, coluna, key_prefix=f"bp_su_{i}")

        col_card_bpu = st.columns([1, 3])[0]
        with col_card_bpu:
            render_card_data_resposta(bp_ultima, titulo="Registro da resposta")

        listas_exist = [c for c in listas_ultima if c in bp_ultima.columns]
        if listas_exist:
            cols_lista_ult = st.columns(min(3, len(listas_exist)))
            for i, coluna in enumerate(listas_exist):
                with cols_lista_ult[i % min(3, len(listas_exist))]:
                    render_lista_respostas(bp_ultima, coluna)

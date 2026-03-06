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

def parse_date_br(x):
    return pd.to_datetime(x, errors="coerce", dayfirst=True)

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

def similaridade_nome(resp_nome: str, cand_nome: str) -> float:
    if not resp_nome or not cand_nome:
        return 0

    if RAPIDFUZZ_OK:
        s1 = fuzz.token_set_ratio(resp_nome, cand_nome)
        s2 = fuzz.token_sort_ratio(resp_nome, cand_nome)
        s3 = fuzz.WRatio(resp_nome, cand_nome)
        s4 = fuzz.partial_ratio(resp_nome, cand_nome)

        score = max(
            s1,
            s2 * 0.98,
            s3 * 0.97,
            s4 * 0.95,
        )

        inter = token_overlap(resp_nome, cand_nome)
        if inter >= 2:
            score += 2
        if primeiro_nome(resp_nome) and primeiro_nome(resp_nome) == primeiro_nome(cand_nome):
            score += 2
        if ult_nome(resp_nome) and ult_nome(resp_nome) == ult_nome(cand_nome):
            score += 1.5

        return min(score, 100)

    base = sequence_ratio(resp_nome, cand_nome)
    inter = token_overlap(resp_nome, cand_nome)
    if inter >= 2:
        base += 3
    if primeiro_nome(resp_nome) == primeiro_nome(cand_nome):
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
        return df.style.applymap(cor_status, subset=["Status"]).set_properties(**{"text-align": "center"})
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

# =========================
# Match de nomes
# =========================
def buscar_melhor_candidato_por_nome(nome_resp, base_lookup, op_resp="", score_min=85):
    if base_lookup.empty or not nome_resp:
        return None, 0

    pool = base_lookup.copy()

    op_resp_norm = norm_text(op_resp)
    if op_resp_norm and "op_norm" in pool.columns:
        pool_op = pool[pool["op_norm"] == op_resp_norm].copy()
        if not pool_op.empty:
            pool = pool_op

    pool = pool.dropna(subset=["nome_norm"]).copy()
    pool = pool[pool["nome_norm"].astype(str).str.strip() != ""].copy()

    if pool.empty:
        return None, 0

    if RAPIDFUZZ_OK:
        candidatos = process.extract(
            nome_resp,
            pool["nome_norm"].tolist(),
            scorer=fuzz.token_set_ratio,
            limit=10,
        )
        nomes_top = [c[0] for c in candidatos] if candidatos else []
        if nomes_top:
            pool = pool[pool["nome_norm"].isin(nomes_top)].copy()

    pool["score_nome"] = pool["nome_norm"].apply(lambda x: similaridade_nome(nome_resp, x))
    pool["token_overlap"] = pool["nome_norm"].apply(lambda x: token_overlap(nome_resp, x))
    pool["primeiro_nome_ok"] = pool["nome_norm"].apply(lambda x: primeiro_nome(nome_resp) == primeiro_nome(x))

    pool = pool.sort_values(
        ["score_nome", "token_overlap", "primeiro_nome_ok"],
        ascending=[False, False, False]
    )

    best = pool.head(1)
    if best.empty:
        return None, 0

    row = best.iloc[0]
    score = float(row["score_nome"])

    if score < score_min:
        return None, score

    if row["token_overlap"] == 0 and score < 92:
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
# Status colaborador (Ativo/Inativo)
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
                score_min=91
            )
            return score

        if not atv_lookup.empty:
            base.loc[falt, "match_ativo_score"] = base.loc[falt].apply(match_ativo, axis=1)
            idx_ok = base.loc[falt].index[base.loc[falt, "match_ativo_score"].fillna(0) >= 91]
            base.loc[idx_ok, "Status Colaborador"] = "Ativo"
            base.loc[idx_ok, "match_ativo_tipo"] = "NOME_FUZZY"

    return base

# =========================
# Vincular respostas NPS/BP à base admitidos
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
            score_min=84
        )
        return cand, score

    falt = nps_m["Colaborador"].isna()
    if falt.any():
        resultados = nps_m.loc[falt].apply(lambda r: fuzzy_match_row(r), axis=1)
        nps_m.loc[falt, "match_score"] = resultados.apply(lambda t: t[1])

        idx_ok = nps_m.loc[falt].index[nps_m.loc[falt, "match_score"].fillna(0) >= 84]
        for idx in idx_ok:
            base_row, _score = fuzzy_match_row(nps_m.loc[idx])
            if base_row is not None:
                for col in ["Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação", "Data", "Data_dt", "Status Colaborador"]:
                    nps_m.at[idx, col] = base_row[col]
                nps_m.at[idx, "match_tipo"] = "NOME_FUZZY"

    falt = bp_m["Colaborador"].isna()
    if falt.any():
        resultados = bp_m.loc[falt].apply(lambda r: fuzzy_match_row(r), axis=1)
        bp_m.loc[falt, "match_score"] = resultados.apply(lambda t: t[1])

        idx_ok = bp_m.loc[falt].index[bp_m.loc[falt, "match_score"].fillna(0) >= 84]
        for idx in idx_ok:
            base_row, _score = fuzzy_match_row(bp_m.loc[idx])
            if base_row is not None:
                for col in ["Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação", "Data", "Data_dt", "Status Colaborador"]:
                    bp_m.at[idx, col] = base_row[col]
                bp_m.at[idx, "match_tipo"] = "NOME_FUZZY"

    return {
        "base_operacional": base,
        "nps_vinculado": nps_m,
        "batepapo_vinculado": bp_m,
    }

# =========================
# FAROL — Etapas
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

            real = (
                form_et.dropna(subset=["Data Cadastro"])
                .dropna(subset=["Colaborador"])
                .groupby("Colaborador", as_index=False)["Data Cadastro"]
                .min()
                .rename(columns={"Data Cadastro": "Data Realização"})
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
        f'<div class="card">'
        f'<h3 style="margin:0; text-align:center;">{titulo}</h3>'
        f'</div>',
        unsafe_allow_html=True
    )

    total = len(df_farol)
    pend_fora = int((df_farol["Status"] == "Não realizado - Fora do prazo").sum())
    pend_atenc_7 = int(
        (
            (df_farol["Status"] == "Não realizado - Atenção") &
            (pd.to_numeric(df_farol["Dias p/ Prazo Máx"], errors="coerce") <= 7)
        ).sum()
    )
    pend_atenc_total = int((df_farol["Status"] == "Não realizado - Atenção").sum())
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
    fig.update_layout(
        height=350,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 100]),
        xaxis_title="",
        yaxis_title="",
        legend_title="Faixa",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_prefix}")

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
        use_container_width=True,
        height=420,
        key=f"df_{key_prefix}"
    )

    excel_bytes = to_excel_bytes(tabela, sheet_name="farol")
    st.download_button(
        "⬇️ Baixar Excel",
        data=excel_bytes,
        file_name=f"farol_padrinhos_{key_prefix.lower()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{key_prefix}",
        use_container_width=True
    )

# =========================
# Respostas - acompanhamento
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

    return df

def identificar_colunas_perguntas(df: pd.DataFrame, origem: str) -> list:
    cols_excluir = {
        "Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação", "Data", "Data_dt",
        "cpf_clean", "nome_norm", "op_norm", "Status Colaborador", "match_tipo",
        "match_score", "Data Cadastro"
    }

    if origem == "NPS":
        cols_excluir.update({
            "Informe seu nome completo:", "Informe seu CPF:", "Informe a operação que você trabalha:",
            "Selecione a semana da avaliação:"
        })
    else:
        cols_excluir.update({
            "Insira o nome do colaborador:", "Inserir o CPF do colaborador:",
            "Selecione a semana do bate papo:"
        })

    colunas = []
    for c in df.columns:
        if c in cols_excluir:
            continue
        serie = df[c]
        if serie.dropna().empty:
            continue
        nunique = serie.dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
        if nunique <= 1:
            continue
        if nunique > 20:
            continue
        colunas.append(c)

    return colunas

def mapa_cores_respostas(valor: str) -> str:
    v = norm_text(valor)
    if v in {"SIM", "SATISFATORIO", "SATISFATÓRIO", "POSITIVO"}:
        return "#79c257"
    if v in {"NAO", "NÃO", "INSATISFATORIO", "INSATISFATÓRIO", "NEGATIVO"}:
        return "#d9534f"
    return "#f0d36b"

def render_graficos_respostas(df_resp: pd.DataFrame, origem: str, key_prefix: str):
    df = filtrar_respostas_por_sidebar(df_resp)
    perguntas = identificar_colunas_perguntas(df, origem=origem)

    if df.empty:
        st.info("Sem respostas para os filtros selecionados.")
        return

    if not perguntas:
        st.info("Não encontrei colunas de respostas válidas para exibir em gráfico.")
        return

    cols = st.columns(4)

    for i, pergunta in enumerate(perguntas):
        base = (
            df[pergunta]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .dropna()
            .value_counts(dropna=False)
            .rename_axis("Resposta")
            .reset_index(name="Qtd")
        )

        if base.empty:
            continue

        total = base["Qtd"].sum()
        base["%"] = (base["Qtd"] / total) * 100
        base["cor"] = base["Resposta"].apply(mapa_cores_respostas)

        fig = px.bar(
            base,
            x="Resposta",
            y="%",
            text=base["%"].round(2).astype(str) + "%",
        )
        fig.update_traces(
            marker_color=base["cor"].tolist(),
            textposition="outside",
            cliponaxis=False
        )
        fig.update_layout(
            height=290,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=60, b=10),
            showlegend=False,
            title=dict(text=pergunta, x=0.5, xanchor="center", font=dict(size=15)),
            yaxis=dict(range=[0, 100], title=""),
            xaxis_title="",
            plot_bgcolor="#4a4a4a",
            paper_bgcolor="#2f2f2f",
        )

        with cols[i % 4]:
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_resp_{i}")

# =========================
# Paths
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARQ_ADMITIDOS = DATA_DIR / "Admitidos.xlsx"
ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"
ARQ_NPS = DATA_DIR / "NPS Mentor.xlsx"
ARQ_BATEPAPO = DATA_DIR / "Bate papo mentor.xlsx"

# =========================
# Carregamento dos arquivos
# =========================
try:
    admitidos = carregar_excel_primeira_aba(ARQ_ADMITIDOS)
    base_ativos = carregar_excel_primeira_aba(ARQ_ATIVOS)
    nps = carregar_excel_primeira_aba(ARQ_NPS)
    batepapo = carregar_excel_primeira_aba(ARQ_BATEPAPO)
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

with st.expander("🔧 Diagnóstico de matching"):
    st.write(f"rapidfuzz instalado? **{RAPIDFUZZ_OK}**")
    st.caption("Para melhorar o reconhecimento de nomes parecidos, vale instalar: pip install rapidfuzz")

    if "match_tipo" in df_nps.columns:
        st.write("**NPS — match por tipo**")
        st.dataframe(
            df_nps["match_tipo"].value_counts(dropna=False).rename_axis("Tipo").reset_index(name="Qtd"),
            use_container_width=True
        )

    if "match_tipo" in df_bp.columns:
        st.write("**Bate-papo — match por tipo**")
        st.dataframe(
            df_bp["match_tipo"].value_counts(dropna=False).rename_axis("Tipo").reset_index(name="Qtd"),
            use_container_width=True
        )

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
    render_graficos_respostas(df_nps, origem="NPS", key_prefix="nps")

with resp_tabs[1]:
    render_graficos_respostas(df_bp, origem="BP", key_prefix="bp")

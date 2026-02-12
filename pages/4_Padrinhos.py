# app_mentor.py
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(page_title="Painel RH - Mentoria", layout="wide")
st.title("🤝 Painel RH — Controle de Padrinhos/Mentores")

# =========================
# Fuzzy (opcional)
# =========================
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# =========================
# Helpers
# =========================
def clean_cpf(x) -> str:
    """CPF só com números. Retorna '' se inválido."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\D", "", s)
    if len(s) == 10:
        s = "0" + s
    return s if len(s) == 11 else ""

def norm_text(x) -> str:
    """Normaliza texto: upper, sem acento, sem pontuação, espaços únicos."""
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

# =========================
# Core pipeline
# =========================
def preparar_base_operacional(admitidos: pd.DataFrame, base_ativos: pd.DataFrame) -> pd.DataFrame:
    """
    - Filtra admitidos a partir de 03/10/2024 (coluna Data em Admitidos)
    - Cruza por Cargo com base_ativos para trazer Tipo Cargo
    - Filtra Tipo Cargo == 'OPERACIONAL LOGÍSTICO'
    """
    adm = admitidos.copy()
    atv = base_ativos.copy()

    # Validar colunas mínimas
    req_adm = ["Colaborador", "CPF", "Cargo", "Data"]
    for c in req_adm:
        if c not in adm.columns:
            raise KeyError(f"Na planilha Admitidos não encontrei a coluna '{c}'")

    req_atv = ["Cargo", "Tipo Cargo"]
    for c in req_atv:
        if c not in atv.columns:
            raise KeyError(f"Na Base colaboradores ativos não encontrei a coluna '{c}'")

    # Datas
    adm["Data_dt"] = parse_date_br(adm["Data"])
    corte = pd.to_datetime("2024-10-03", dayfirst=True)
    adm = adm[adm["Data_dt"] >= corte].copy()

    # Normalizações
    adm["cpf_clean"] = adm["CPF"].apply(clean_cpf)
    adm["nome_norm"] = adm["Colaborador"].apply(norm_text)
    adm["cargo_norm"] = adm["Cargo"].apply(norm_text)

    atv["cargo_norm"] = atv["Cargo"].apply(norm_text)

    # 1) Merge direto por cargo_norm
    merged = adm.merge(
        atv[["cargo_norm", "Tipo Cargo"]].drop_duplicates(),
        on="cargo_norm",
        how="left",
    )

    # 2) Fuzzy de Cargo (opcional) para preencher Tipo Cargo faltante
    merged["cargo_match_score"] = pd.NA
    if RAPIDFUZZ_OK:
        faltantes = merged["Tipo Cargo"].isna()
        if faltantes.any():
            cargos_ref = (
                atv[["cargo_norm", "Tipo Cargo"]]
                .dropna(subset=["cargo_norm", "Tipo Cargo"])
                .drop_duplicates()
            )
            ref_list = cargos_ref["cargo_norm"].tolist()
            ref_map = dict(zip(cargos_ref["cargo_norm"], cargos_ref["Tipo Cargo"]))

            def fuzzy_tipo_cargo(cargo_norm):
                if not cargo_norm:
                    return None, 0
                hit = process.extractOne(cargo_norm, ref_list, scorer=fuzz.WRatio)
                if not hit:
                    return None, 0
                best, score, _ = hit
                return ref_map.get(best), score

            tipo_score = merged.loc[faltantes, "cargo_norm"].apply(fuzzy_tipo_cargo)
            merged.loc[faltantes, "Tipo Cargo"] = tipo_score.apply(lambda t: t[0])
            merged.loc[faltantes, "cargo_match_score"] = tipo_score.apply(lambda t: t[1])

    merged["Tipo Cargo"] = merged["Tipo Cargo"].fillna("")
    oper = merged[merged["Tipo Cargo"].str.upper().eq("OPERACIONAL LOGÍSTICO")].copy()
    return oper

def vincular_checks(base_oper: pd.DataFrame, nps: pd.DataFrame, batepapo: pd.DataFrame) -> dict:
    """
    Vínculo por CPF (principal) + fallback por nome (fuzzy opcional).
    """
    base = base_oper.copy()
    base["cpf_clean"] = base["cpf_clean"].fillna("")
    base["nome_norm"] = base["nome_norm"].fillna("")

    # ---- NPS Mentor ----
    nps_df = nps.copy()
    # colunas conforme você passou
    nps_nome_col = "Informe seu nome completo:"
    nps_cpf_col = "Informe seu CPF:"
    nps_op_col = "Informe a operação que você trabalha:"  # se existir

    for c in [nps_nome_col, nps_cpf_col]:
        if c not in nps_df.columns:
            raise KeyError(f"No NPS Mentor não encontrei a coluna '{c}'")

    nps_df["cpf_clean"] = nps_df[nps_cpf_col].apply(clean_cpf)
    nps_df["nome_norm"] = nps_df[nps_nome_col].apply(norm_text)
    nps_df["op_norm"] = nps_df[nps_op_col].apply(norm_text) if nps_op_col in nps_df.columns else ""

    # ---- Bate papo mentor ----
    bp = batepapo.copy()
    bp_nome_col = "Insira o nome do colaborador:"
    bp_cpf_col = "Inserir o CPF do colaborador:"

    for c in [bp_nome_col, bp_cpf_col]:
        if c not in bp.columns:
            raise KeyError(f"No Bate papo mentor não encontrei a coluna '{c}'")

    bp["cpf_clean"] = bp[bp_cpf_col].apply(clean_cpf)
    bp["nome_norm"] = bp[bp_nome_col].apply(norm_text)
    bp["op_norm"] = ""  # se um dia tiver coluna de operação aqui, normaliza igual ao NPS

    # Se base tiver Operação, use como trava (opcional)
    base["op_norm"] = base["Operação"].apply(norm_text) if "Operação" in base.columns else ""

    # ================
    # Match por CPF
    # ================
    base_cols = ["cpf_clean", "Colaborador", "CPF", "Cargo", "Tipo Cargo", "Data", "Data_dt", "nome_norm", "op_norm"]
    nps_m = nps_df.merge(base[base_cols], on="cpf_clean", how="left", suffixes=("", "_base"))
    bp_m = bp.merge(base[base_cols], on="cpf_clean", how="left", suffixes=("", "_base"))

    nps_m["match_tipo"] = nps_m["Colaborador"].apply(lambda x: "CPF" if pd.notna(x) else "NAO_ENCONTRADO")
    bp_m["match_tipo"] = bp_m["Colaborador"].apply(lambda x: "CPF" if pd.notna(x) else "NAO_ENCONTRADO")

    nps_m["match_score"] = pd.NA
    bp_m["match_score"] = pd.NA

    # ================
    # Fallback por NOME (fuzzy)
    # ================
    if RAPIDFUZZ_OK:
        base_lookup = base.copy()

        def fuzzy_match_nome(row):
            nome = row.get("nome_norm", "")
            if not nome:
                return None, 0

            op_form = row.get("op_norm", "")
            if op_form and "op_norm" in base_lookup.columns:
                pool = base_lookup[base_lookup["op_norm"].eq(op_form)]
                if pool.empty:
                    pool = base_lookup
            else:
                pool = base_lookup

            choices = pool["nome_norm"].dropna().unique().tolist()
            hit = process.extractOne(nome, choices, scorer=fuzz.WRatio)
            if not hit:
                return None, 0

            best, score, _ = hit
            base_row = pool[pool["nome_norm"] == best].head(1)
            if base_row.empty:
                return None, score
            return base_row.iloc[0], score

        # NPS
        falt_nps = nps_m["Colaborador"].isna()
        if falt_nps.any():
            out = nps_m.loc[falt_nps].apply(lambda r: fuzzy_match_nome(r), axis=1)
            nps_m.loc[falt_nps, "match_score"] = out.apply(lambda t: t[1])

            # aceita automático se >= 90
            idx_ok = nps_m.loc[falt_nps].index[(nps_m.loc[falt_nps, "match_score"].fillna(0) >= 90)]
            for idx in idx_ok:
                base_row, score = fuzzy_match_nome(nps_m.loc[idx])
                if base_row is not None:
                    for col in ["Colaborador", "CPF", "Cargo", "Tipo Cargo", "Data", "Data_dt"]:
                        nps_m.at[idx, col] = base_row[col]
                    nps_m.at[idx, "match_tipo"] = "NOME_FUZZY"

        # Bate-papo
        falt_bp = bp_m["Colaborador"].isna()
        if falt_bp.any():
            out = bp_m.loc[falt_bp].apply(lambda r: fuzzy_match_nome(r), axis=1)
            bp_m.loc[falt_bp, "match_score"] = out.apply(lambda t: t[1])

            idx_ok = bp_m.loc[falt_bp].index[(bp_m.loc[falt_bp, "match_score"].fillna(0) >= 90)]
            for idx in idx_ok:
                base_row, score = fuzzy_match_nome(bp_m.loc[idx])
                if base_row is not None:
                    for col in ["Colaborador", "CPF", "Cargo", "Tipo Cargo", "Data", "Data_dt"]:
                        bp_m.at[idx, col] = base_row[col]
                    bp_m.at[idx, "match_tipo"] = "NOME_FUZZY"

    pend_nps = nps_m[nps_m["Colaborador"].isna()].copy()
    pend_bp = bp_m[bp_m["Colaborador"].isna()].copy()

    return {
        "base_operacional": base,
        "nps_vinculado": nps_m,
        "batepapo_vinculado": bp_m,
        "pendencias_nps": pend_nps,
        "pendencias_batepapo": pend_bp,
    }

# =========================
# Carregamento de arquivos
# (AJUSTE AQUI pro seu projeto)
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# 👉 Ajuste os nomes dos arquivos
ARQ_ADMITIDOS = DATA_DIR / "Admitidos.xlsx"
ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"
ARQ_NPS = DATA_DIR / "NPS Mentor.xlsx"
ARQ_BATEPAPO = DATA_DIR / "Bate papo mentor.xlsx"

@st.cache_data(show_spinner=True)
def carregar_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # sempre pega a primeira aba (independente do nome)
    xls = pd.ExcelFile(path)
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

try:
    admitidos = carregar_excel(ARQ_ADMITIDOS)
    base_ativos = carregar_excel(ARQ_ATIVOS)
    nps = carregar_excel(ARQ_NPS)
    batepapo = carregar_excel(ARQ_BATEPAPO)

except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}")
    st.stop()


# =========================
# Processamento
# =========================
try:
    base_oper = preparar_base_operacional(admitidos, base_ativos)
    result = vincular_checks(base_oper, nps, batepapo)

    df_nps = result["nps_vinculado"]
    df_bp = result["batepapo_vinculado"]
    pend_nps = result["pendencias_nps"]
    pend_bp = result["pendencias_batepapo"]
except Exception as e:
    st.error(f"Erro no processamento: {e}")
    st.stop()

# =========================
# UI — Resumo e Tabelas
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Admitidos (>= 03/10/2024)", f"{len(admitidos):,}".replace(",", "."))
col2.metric("Operacional Logístico (filtrado)", f"{len(base_oper):,}".replace(",", "."))
col3.metric("NPS (linhas)", f"{len(df_nps):,}".replace(",", "."))
col4.metric("Bate-papo (linhas)", f"{len(df_bp):,}".replace(",", "."))

st.divider()

cA, cB = st.columns(2)
with cA:
    st.subheader("✅ NPS Mentor — Vinculado")
    st.caption("Match por CPF (prioridade) + fallback por nome (se rapidfuzz estiver instalado).")
    st.dataframe(df_nps, use_container_width=True)

with cB:
    st.subheader("✅ Bate-papo mentor — Vinculado")
    st.dataframe(df_bp, use_container_width=True)

st.divider()

cC, cD = st.columns(2)
with cC:
    st.subheader("⚠️ Pendências — NPS (não encontrado na base operacional)")
    st.caption("Se aparecer muita coisa aqui, normalmente é CPF digitado errado ou nome muito diferente.")
    st.dataframe(pend_nps, use_container_width=True)

with cD:
    st.subheader("⚠️ Pendências — Bate-papo (não encontrado na base operacional)")
    st.dataframe(pend_bp, use_container_width=True)

st.divider()

with st.expander("🧩 Diagnóstico rápido"):
    st.write(f"rapidfuzz instalado? **{RAPIDFUZZ_OK}**")
    if not RAPIDFUZZ_OK:
        st.info("Se você quiser o match aproximado por nome, instale: pip install rapidfuzz")
    st.write("Exemplos de colunas usadas:")
    st.code(
        "Admitidos: Colaborador | CPF | Cargo | Data\n"
        "Base ativos: Cargo | Tipo Cargo\n"
        "NPS: Informe seu nome completo: | Informe seu CPF: | (opcional) Informe a operação que você trabalha:\n"
        "Bate-papo: Insira o nome do colaborador: | Inserir o CPF do colaborador:",
        language="text",
    )

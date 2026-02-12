import re
import unicodedata
import pandas as pd

# Fuzzy opcional (recomendado). Se não tiver instalado, o código segue sem fuzzy.
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False


# =========================
# Normalizadores
# =========================
def clean_cpf(x) -> str:
    """Retorna CPF só com números (11 dígitos quando possível)."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\D", "", s)  # só números
    # alguns digitam com 10 dígitos e esquecem zero à esquerda
    if len(s) == 10:
        s = "0" + s
    return s if len(s) == 11 else ""


def norm_text(x) -> str:
    """Normaliza texto: maiúsculo, sem acento, sem pontuação, espaços únicos."""
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9\s]", " ", s)  # remove pontuação
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_date_br(x):
    """Tenta converter datas (excel/str) para datetime."""
    return pd.to_datetime(x, errors="coerce", dayfirst=True)


# =========================
# Carregamento (ajuste para seus paths)
# =========================
# Exemplo:
# admitidos = pd.read_excel(ARQ_ADMITIDOS, sheet_name="Admitidos")
# nps = pd.read_excel(ARQ_NPS, sheet_name="NPS Mentor")
# batepapo = pd.read_excel(ARQ_BATEPAPO, sheet_name="Bate papo mentor")
# base_ativos = pd.read_excel(ARQ_ATIVOS, sheet_name="Base colaboradores ativos")


# =========================
# Pipeline
# =========================
def preparar_base_operacional(admitidos: pd.DataFrame, base_ativos: pd.DataFrame) -> pd.DataFrame:
    """
    - Filtra admitidos a partir de 03/10/2024 (coluna Data)
    - Cruza por Cargo com base_ativos para trazer Tipo Cargo
    - Filtra Tipo Cargo == 'OPERACIONAL LOGÍSTICO'
    """
    adm = admitidos.copy()
    atv = base_ativos.copy()

    # Datas
    adm["Data_dt"] = parse_date_br(adm["Data"])
    corte = pd.to_datetime("2024-10-03", dayfirst=True)
    adm = adm[adm["Data_dt"] >= corte].copy()

    # Chaves e normalizações
    adm["cpf_clean"] = adm["CPF"].apply(clean_cpf)
    adm["nome_norm"] = adm["Colaborador"].apply(norm_text)
    adm["cargo_norm"] = adm["Cargo"].apply(norm_text)

    # Base ativos: precisamos de Cargo e Tipo Cargo
    atv["cargo_norm"] = atv["Cargo"].apply(norm_text)
    # Se houver "Tipo Cargo" como coluna
    if "Tipo Cargo" not in atv.columns:
        raise KeyError("Na Base colaboradores ativos não encontrei a coluna 'Tipo Cargo'.")

    # 1) Merge direto por cargo_norm
    merged = adm.merge(
        atv[["cargo_norm", "Tipo Cargo"]].drop_duplicates(),
        on="cargo_norm",
        how="left",
        suffixes=("", "_ativos"),
    )

    # 2) Se tiver cargos não casados e rapidfuzz estiver disponível, tenta fuzzy de cargo
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
    else:
        merged["cargo_match_score"] = None

    # Filtro final operacional logístico
    merged["Tipo Cargo"] = merged["Tipo Cargo"].fillna("")
    oper = merged[merged["Tipo Cargo"].str.upper().eq("OPERACIONAL LOGÍSTICO")].copy()

    return oper


def vincular_checks(
    base_oper: pd.DataFrame,
    nps: pd.DataFrame,
    batepapo: pd.DataFrame,
) -> dict:
    """
    Vínculo por CPF (principal).
    Fallback por nome_norm (opcional) e, se existir, Operação como trava.
    Retorna dict com tabelas prontas (base + nps + batepapo + pendências).
    """

    # ---- NPS ----
    nps_df = nps.copy()
    nps_df["cpf_clean"] = nps_df["Informe seu CPF:"].apply(clean_cpf)
    nps_df["nome_norm"] = nps_df["Informe seu nome completo:"].apply(norm_text)

    # Se existir operação no NPS, vamos normalizar pra usar de trava
    op_col_nps = "Informe a operação que você trabalha:"
    if op_col_nps in nps_df.columns:
        nps_df["op_norm"] = nps_df[op_col_nps].apply(norm_text)
    else:
        nps_df["op_norm"] = ""

    # ---- Bate-papo ----
    bp = batepapo.copy()
    bp["cpf_clean"] = bp["Inserir o CPF do colaborador:"].apply(clean_cpf)
    bp["nome_norm"] = bp["Insira o nome do colaborador:"].apply(norm_text)

    # (se existir operação aqui também, ajuste o nome da coluna e normaliza)
    # bp["op_norm"] = bp["SUA_COLUNA_DE_OPERACAO"].apply(norm_text) if "SUA_COLUNA_DE_OPERACAO" in bp.columns else ""

    # ---- Base operacional ----
    base = base_oper.copy()
    base["cpf_clean"] = base["cpf_clean"].fillna("")
    base["nome_norm"] = base["nome_norm"].fillna("")

    # Se base tiver Operação, dá pra travar melhor o fuzzy. Se não tiver, fica vazio.
    base["op_norm"] = base["Operação"].apply(norm_text) if "Operação" in base.columns else ""

    # ================
    # 1) Match por CPF
    # ================
    nps_m = nps_df.merge(
        base[["cpf_clean", "Colaborador", "CPF", "Cargo", "Tipo Cargo", "Data", "Data_dt", "nome_norm", "op_norm"]],
        on="cpf_clean",
        how="left",
        suffixes=("", "_base"),
    )
    nps_m["match_tipo"] = nps_m["Colaborador"].apply(lambda x: "CPF" if pd.notna(x) else "NAO_ENCONTRADO")

    bp_m = bp.merge(
        base[["cpf_clean", "Colaborador", "CPF", "Cargo", "Tipo Cargo", "Data", "Data_dt", "nome_norm", "op_norm"]],
        on="cpf_clean",
        how="left",
        suffixes=("", "_base"),
    )
    bp_m["match_tipo"] = bp_m["Colaborador"].apply(lambda x: "CPF" if pd.notna(x) else "NAO_ENCONTRADO")

    # =========================================
    # 2) Fallback fuzzy por nome (se disponível)
    # =========================================
    if RAPIDFUZZ_OK:
        # Preparar lookup por operação (quando existir)
        base_lookup = base.copy()

        def fuzzy_match_nome(row, base_df):
            nome = row.get("nome_norm", "")
            if not nome:
                return None, 0
            # trava por operação se existir nas duas pontas
            op_form = row.get("op_norm", "")
            if op_form and "op_norm" in base_df.columns:
                pool = base_df[base_df["op_norm"].eq(op_form)]
                if pool.empty:
                    pool = base_df
            else:
                pool = base_df

            choices = pool["nome_norm"].dropna().unique().tolist()
            hit = process.extractOne(nome, choices, scorer=fuzz.WRatio)
            if not hit:
                return None, 0
            best, score, _ = hit
            # retorna a linha base correspondente ao melhor nome
            base_row = pool[pool["nome_norm"] == best].head(1)
            if base_row.empty:
                return None, score
            return base_row.iloc[0], score

        # NPS
        falt_nps = nps_m["Colaborador"].isna()
        if falt_nps.any():
            out = nps_m.loc[falt_nps].apply(lambda r: fuzzy_match_nome(r, base_lookup), axis=1)
            nps_m.loc[falt_nps, "match_score"] = out.apply(lambda t: t[1])
            # aceita automático se score >= 90
            ok = (nps_m.loc[falt_nps, "match_score"].fillna(0) >= 90)
            idx_ok = nps_m.loc[falt_nps].index[ok]

            # preencher campos da base quando aceito
            for idx in idx_ok:
                base_row, score = fuzzy_match_nome(nps_m.loc[idx], base_lookup)
                if base_row is not None:
                    nps_m.at[idx, "Colaborador"] = base_row["Colaborador"]
                    nps_m.at[idx, "CPF"] = base_row["CPF"]
                    nps_m.at[idx, "Cargo"] = base_row["Cargo"]
                    nps_m.at[idx, "Tipo Cargo"] = base_row["Tipo Cargo"]
                    nps_m.at[idx, "Data"] = base_row["Data"]
                    nps_m.at[idx, "Data_dt"] = base_row["Data_dt"]
                    nps_m.at[idx, "match_tipo"] = "NOME_FUZZY"

        # Bate-papo
        falt_bp = bp_m["Colaborador"].isna()
        if falt_bp.any():
            out = bp_m.loc[falt_bp].apply(lambda r: fuzzy_match_nome(r, base_lookup), axis=1)
            bp_m.loc[falt_bp, "match_score"] = out.apply(lambda t: t[1])
            ok = (bp_m.loc[falt_bp, "match_score"].fillna(0) >= 90)
            idx_ok = bp_m.loc[falt_bp].index[ok]

            for idx in idx_ok:
                base_row, score = fuzzy_match_nome(bp_m.loc[idx], base_lookup)
                if base_row is not None:
                    bp_m.at[idx, "Colaborador"] = base_row["Colaborador"]
                    bp_m.at[idx, "CPF"] = base_row["CPF"]
                    bp_m.at[idx, "Cargo"] = base_row["Cargo"]
                    bp_m.at[idx, "Tipo Cargo"] = base_row["Tipo Cargo"]
                    bp_m.at[idx, "Data"] = base_row["Data"]
                    bp_m.at[idx, "Data_dt"] = base_row["Data_dt"]
                    bp_m.at[idx, "match_tipo"] = "NOME_FUZZY"

    # Pendências (não encontrou na base operacional mesmo após tentativa)
    pend_nps = nps_m[nps_m["Colaborador"].isna()].copy()
    pend_bp = bp_m[bp_m["Colaborador"].isna()].copy()

    return {
        "base_operacional": base,
        "nps_vinculado": nps_m,
        "batepapo_vinculado": bp_m,
        "pendencias_nps": pend_nps,
        "pendencias_batepapo": pend_bp,
    }

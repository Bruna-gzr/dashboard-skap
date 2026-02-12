# pages/4_Padrinhos.py
# Versão COMPLETA corrigida:
# - Sem IndentationError
# - Sem StreamlitDuplicateElementId (keys únicas em gráficos/tabelas)
# - Aderência nova: OK = realizado (mesmo fora do prazo) + pendente ainda no prazo | RUIM = pendente fora do prazo
# - Datas em dd/mm/aaaa (abreviado)
# - Filtros: Operação / Data (admissão) / Cargo / Status

import re
import unicodedata
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# (Opcional) Fuzzy match por nome/cargo
# =========================
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# =========================
# Page config
# =========================
st.set_page_config(page_title="Painel RH — Mentoria", layout="wide")
st.title("🤝 Painel RH — Mentoria / Padrinhos")

# =========================
# Estilo (dark + dourado)
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
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers (limpeza)
# =========================
def clean_cpf(x) -> str:
    """CPF só com números (11 dígitos). Retorna '' se inválido."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\D", "", s)
    if len(s) == 10:
        s = "0" + s
    return s if len(s) == 11 else ""

def norm_text(x) -> str:
    """Upper, sem acento, sem pontuação, espaços únicos."""
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
# Carregamento Excel (primeira aba)
# =========================
@st.cache_data(show_spinner=True)
def carregar_excel_primeira_aba(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    xls = pd.ExcelFile(path)
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

# =========================
# Pipeline: Admitidos -> cruza Tipo Cargo -> filtra operacional
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

    adm["Data_dt"] = parse_date_br(adm["Data"])
    corte = pd.to_datetime("2024-10-03", dayfirst=True)
    adm = adm[adm["Data_dt"] >= corte].copy()

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
    return oper

# =========================
# Pipeline: vincular NPS/Bate-papo (CPF + fuzzy nome opcional)
# =========================
def vincular_checks(base_oper: pd.DataFrame, nps: pd.DataFrame, batepapo: pd.DataFrame) -> dict:
    base = base_oper.copy()
    base["cpf_clean"] = base["cpf_clean"].fillna("")
    base["nome_norm"] = base["nome_norm"].fillna("")
    base["op_norm"] = base["Operação"].apply(norm_text) if "Operação" in base.columns else ""

    # ---- NPS Mentor ----
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

    # ---- Bate-papo mentor ----
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

    base_cols = ["cpf_clean", "Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação", "Data", "Data_dt", "nome_norm", "op_norm"]
    nps_m = nps_df.merge(base[base_cols], on="cpf_clean", how="left", suffixes=("", "_base"))
    bp_m = bp.merge(base[base_cols], on="cpf_clean", how="left", suffixes=("", "_base"))

    nps_m["match_tipo"] = nps_m["Colaborador"].apply(lambda x: "CPF" if pd.notna(x) else "NAO_ENCONTRADO")
    bp_m["match_tipo"] = bp_m["Colaborador"].apply(lambda x: "CPF" if pd.notna(x) else "NAO_ENCONTRADO")
    nps_m["match_score"] = pd.NA
    bp_m["match_score"] = pd.NA

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

        # NPS fallback
        falt = nps_m["Colaborador"].isna()
        if falt.any():
            out = nps_m.loc[falt].apply(lambda r: fuzzy_match_nome(r), axis=1)
            nps_m.loc[falt, "match_score"] = out.apply(lambda t: t[1])
            idx_ok = nps_m.loc[falt].index[(nps_m.loc[falt, "match_score"].fillna(0) >= 90)]
            for idx in idx_ok:
                base_row, _score = fuzzy_match_nome(nps_m.loc[idx])
                if base_row is not None:
                    for col in ["Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação", "Data", "Data_dt"]:
                        nps_m.at[idx, col] = base_row[col]
                    nps_m.at[idx, "match_tipo"] = "NOME_FUZZY"

        # BP fallback
        falt = bp_m["Colaborador"].isna()
        if falt.any():
            out = bp_m.loc[falt].apply(lambda r: fuzzy_match_nome(r), axis=1)
            bp_m.loc[falt, "match_score"] = out.apply(lambda t: t[1])
            idx_ok = bp_m.loc[falt].index[(bp_m.loc[falt, "match_score"].fillna(0) >= 90)]
            for idx in idx_ok:
                base_row, _score = fuzzy_match_nome(bp_m.loc[idx])
                if base_row is not None:
                    for col in ["Colaborador", "CPF", "Cargo", "Tipo Cargo", "Operação", "Data", "Data_dt"]:
                        bp_m.at[idx, col] = base_row[col]
                    bp_m.at[idx, "match_tipo"] = "NOME_FUZZY"

    return {
        "base_operacional": base,
        "nps_vinculado": nps_m,
        "batepapo_vinculado": bp_m,
    }

# =========================
# FAROL — prazos
# =========================
ETAPAS = [
    {"chave": "NPS_1_SEMANA", "titulo": "NPS 1ª SEMANA", "tipo": "NPS",
     "campo_selecao": "Selecione a semana da avaliação:", "valor_selecao": "Primeira semana junto ao padrinho.",
     "prazo_min_dias": 11, "prazo_max_dias": 14},

    {"chave": "NPS_ULTIMA", "titulo": "NPS ÚLTIMA SEMANA", "tipo": "NPS",
     "campo_selecao": "Selecione a semana da avaliação:", "valor_selecao": "Última semana junto ao padrinho.",
     "prazo_min_dias": 20, "prazo_max_dias": 32},

    {"chave": "BP_2_SEMANA", "titulo": "BATE-PAPO — 2ª SEMANA", "tipo": "BP",
     "campo_selecao": "Selecione a semana do bate papo:", "valor_selecao": "Segunda Semana",
     "prazo_min_dias": 11, "prazo_max_dias": 14},

    {"chave": "BP_3_SEMANA", "titulo": "BATE-PAPO — 3ª SEMANA", "tipo": "BP",
     "campo_selecao": "Selecione a semana do bate papo:", "valor_selecao": "Terceira Semana",
     "prazo_min_dias": 20, "prazo_max_dias": 22},

    {"chave": "BP_ULTIMA", "titulo": "BATE-PAPO — ÚLTIMA SEMANA", "tipo": "BP",
     "campo_selecao": "Selecione a semana do bate papo:", "valor_selecao": "Última Semana",
     "prazo_min_dias": 28, "prazo_max_dias": 32},
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

def montar_farol_por_etapa(base_oper, df_nps, df_bp, hoje):
    base = base_oper.copy()
    if "Data_dt" not in base.columns:
        base["Data_dt"] = pd.to_datetime(base["Data"], errors="coerce", dayfirst=True)

    farois = {}
    for etapa in ETAPAS:
        tmp = base[["Colaborador", "CPF", "cpf_clean", "Operação", "Cargo", "Data_dt"]].copy()
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
                .groupby("cpf_clean", as_index=False)["Data Cadastro"]
                .min()
                .rename(columns={"Data Cadastro": "Data Realização"})
            )
            tmp = tmp.merge(real, on="cpf_clean", how="left")

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

def formatar_datas_para_tabela(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Data Admissão", "Prazo Mín", "Prazo Máx", "Data Realização"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", dayfirst=True).dt.strftime("%d/%m/%Y")
    return out

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
        f'<div class="small-muted" style="text-align:center;">Aderência por operação + lista de pendências</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    total = len(df_farol)
    pend_fora = int((df_farol["Status"] == "Não realizado - Fora do prazo").sum())
    pend_atenc = int((df_farol["Status"] == "Não realizado - Atenção").sum())
    realizados = int(df_farol["Status"].isin(["Realizado no prazo", "Realizado fora do prazo", "Realizado antes do prazo"]).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{total:,}".replace(",", "."), key=f"m_total_{key_prefix}")
    c2.metric("Pend. fora do prazo", f"{pend_fora:,}".replace(",", "."), key=f"m_fora_{key_prefix}")
    c3.metric("Pend. atenção (no prazo)", f"{pend_atenc:,}".replace(",", "."), key=f"m_atenc_{key_prefix}")
    c4.metric("Realizados", f"{realizados:,}".replace(",", "."), key=f"m_real_{key_prefix}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    g = (
        df_farol.assign(pend_fora=(df_farol["Status"] == "Não realizado - Fora do prazo"))
        .groupby("Operação", as_index=False)
        .agg(total=("Colaborador", "count"), pend_fora=("pend_fora", "sum"))
    )
    g["Aderência %"] = ((g["total"] - g["pend_fora"]) / g["total"]).fillna(0) * 100
    g = g.sort_values("Aderência %", ascending=False)

    fig = px.bar(
        g,
        x="Operação",
        y="Aderência %",
        text=g["Aderência %"].round(2).astype(str) + "%",
    )
    fig.update_layout(
        height=330,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 100]),
        xaxis_title="",
        yaxis_title="",
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_color="#f0d36b")
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_prefix}")

    st.markdown(
        '<div class="card"><h4 style="margin:0; text-align:center;">LISTA — PENDÊNCIAS PARA COBRANÇA</h4></div>',
        unsafe_allow_html=True
    )

    pend = df_farol[df_farol["Status"].isin(["Não realizado - Fora do prazo", "Não realizado - Atenção"])].copy()

    cols_show = [
        "Operação", "Colaborador", "CPF", "Cargo",
        "Data_dt", "Prazo Mín", "Prazo Máx", "Data Realização",
        "Dias p/ Prazo Máx", "Status"
    ]
    cols_show = [c for c in cols_show if c in pend.columns]
    pend = pend[cols_show].rename(columns={"Data_dt": "Data Admissão"})

    pend = formatar_datas_para_tabela(pend)
    st.dataframe(style_table(pend), use_container_width=True, height=340, key=f"df_{key_prefix}")

# =========================
# Paths (AJUSTE OS NOMES)
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARQ_ADMITIDOS = DATA_DIR / "Admitidos.xlsx"
ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"
ARQ_NPS = DATA_DIR / "NPS Mentor.xlsx"
ARQ_BATEPAPO = DATA_DIR / "Bate papo mentor.xlsx"

# =========================
# Carregar arquivos
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
# Pipeline completo
# =========================
try:
    base_oper = preparar_base_operacional(admitidos, base_ativos)
    result = vincular_checks(base_oper, nps, batepapo)
    df_nps = result["nps_vinculado"]
    df_bp = result["batepapo_vinculado"]
except Exception as e:
    st.error(f"Erro no pipeline de mentoria: {e}")
    st.stop()

# =========================
# Top metrics
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Admitidos (arquivo)", f"{len(admitidos):,}".replace(",", "."))
c2.metric("Operacional Logístico (>=03/10/2024)", f"{len(base_oper):,}".replace(",", "."))
c3.metric("NPS (linhas)", f"{len(df_nps):,}".replace(",", "."))
c4.metric("Bate-papo (linhas)", f"{len(df_bp):,}".replace(",", "."))

with st.expander("🔧 Diagnóstico"):
    st.write(f"rapidfuzz instalado? **{RAPIDFUZZ_OK}** (opcional, para match por nome/cargo)")
    st.caption("Se quiser ativar o match aproximado, instale no ambiente: pip install rapidfuzz")

# =========================
# FAROL (com filtros)
# =========================
st.header("🚦 ADERÊNCIA — PROCESSO PADRINHOS (FAROL)")
st.subheader("🔎 Filtros")

if "Data_dt" not in base_oper.columns:
    base_oper["Data_dt"] = pd.to_datetime(base_oper["Data"], errors="coerce", dayfirst=True)

ops_all = sorted([x for x in base_oper["Operação"].fillna("").astype(str).unique().tolist() if x.strip()])
cargos_all = sorted([x for x in base_oper["Cargo"].fillna("").astype(str).unique().tolist() if x.strip()])

status_options = [
    "Não realizado - Fora do prazo",
    "Não realizado - Atenção",
    "Realizado fora do prazo",
    "Realizado no prazo",
    "Realizado antes do prazo",
]

colf1, colf2, colf3, colf4 = st.columns([2, 2, 2, 2])

with colf1:
    filtro_ops = st.multiselect("Operação", options=ops_all, default=[], key="f_ops")

with colf2:
    data_min = pd.to_datetime(base_oper["Data_dt"], errors="coerce").min()
    data_max = pd.to_datetime(base_oper["Data_dt"], errors="coerce").max()
    if pd.isna(data_min):
        data_min = pd.Timestamp("2024-10-03")
    if pd.isna(data_max):
        data_max = pd.Timestamp(datetime.now().date())

    dt_ini, dt_fim = st.date_input(
        "Data de admissão (intervalo)",
        value=(data_min.date(), data_max.date()),
        key="f_dt",
    )
    dt_ini = pd.Timestamp(dt_ini)
    dt_fim = pd.Timestamp(dt_fim)

with colf3:
    filtro_cargos = st.multiselect("Cargo", options=cargos_all, default=[], key="f_cargo")

with colf4:
    filtro_status = st.multiselect("Status", options=status_options, default=[], key="f_status")

def aplicar_filtros_farol(df_farol: pd.DataFrame) -> pd.DataFrame:
    df = df_farol.copy()

    if filtro_ops:
        df = df[df["Operação"].isin(filtro_ops)]

    if "Data_dt" in df.columns:
        df = df[(df["Data_dt"] >= dt_ini) & (df["Data_dt"] <= dt_fim)]

    if filtro_cargos:
        df = df[df["Cargo"].isin(filtro_cargos)]

    if filtro_status:
        df = df[df["Status"].isin(filtro_status)]

    return df

# ---------- Executa farol ----------
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

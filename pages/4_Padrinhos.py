# pages/4_Padrinhos.py
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Importar funções do app.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import get_operacao

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
# IMPORTS
# =========================
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
import plotly.express as px

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# =========================
# CSS
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
# FUNÇÕES AUXILIARES
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
# FUNÇÕES DO PIPELINE
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
    adm["nome_norm_flex"] = adm["Colaborador"].apply(norm_text_nome_flex)
    adm["cargo_norm"] = adm["Cargo"].apply(norm_text)
    adm["op_norm"] = adm["Operação"].apply(norm_text)

    atv["cargo_norm"] = atv["Cargo"].apply(norm_text)

    merged = adm.merge(
        atv[["cargo_norm", "Tipo Cargo"]].drop_duplicates(),
        on="cargo_norm",
        how="left",
    )

    merged["Tipo Cargo"] = merged["Tipo Cargo"].fillna("")
    oper = merged[merged["Tipo Cargo"].str.upper().eq("OPERACIONAL LOGÍSTICO")].copy()

    oper = oper[oper["Data_dt"] >= pd.Timestamp("2024-10-03")].copy()

    oper["Operação"] = oper["Operação"].astype(str).str.strip()
    oper["op_norm"] = oper["Operação"].apply(norm_text)

    oper["pessoa_key"] = oper.apply(
        lambda r: r["cpf_clean"] if str(r.get("cpf_clean", "")).strip() else r["nome_norm_flex"],
        axis=1
    )

    oper = (
        oper.sort_values(["pessoa_key", "Data_dt"], ascending=[True, False])
            .drop_duplicates(subset=["pessoa_key"], keep="first")
            .copy()
    )

    oper = oper.sort_values(["nome_norm_flex", "cpf_clean", "Data_dt"]).reset_index(drop=True)
    oper["contrato_id"] = oper.index.astype(str)

    return oper

def classificar_status_colaborador(base_oper: pd.DataFrame, base_ativos: pd.DataFrame) -> pd.DataFrame:
    base = base_oper.copy()
    atv = base_ativos.copy()

    if "Colaborador" not in atv.columns:
        raise KeyError("Na Base colaboradores ativos não encontrei a coluna 'Colaborador'")

    atv["nome_norm"] = atv["Colaborador"].apply(norm_text)
    atv["nome_norm_flex"] = atv["Colaborador"].apply(norm_text_nome_flex)
    atv["op_norm"] = atv["Operação"].apply(norm_text) if "Operação" in atv.columns else ""

    col_cpf_ativos = None
    for c in atv.columns:
        if norm_text(c) == "CPF":
            col_cpf_ativos = c
            break

    if col_cpf_ativos is not None:
        atv["cpf_clean"] = atv[col_cpf_ativos].apply(clean_cpf)
    else:
        atv["cpf_clean"] = ""

    ativos_lookup = (
        atv[["cpf_clean", "nome_norm", "nome_norm_flex", "op_norm"]]
        .drop_duplicates()
        .copy()
    )

    status_lista = []
    for _, row in base.iterrows():
        nome_base = row.get("Colaborador", "")
        nome_base_flex = norm_text_nome_flex(nome_base)
        cpf_base = str(row.get("cpf_clean", "")).strip()
        op_base = norm_text(row.get("Operação", ""))

        pool = ativos_lookup.copy()

        if op_base and "op_norm" in pool.columns:
            pool_op = pool[pool["op_norm"] == op_base].copy()
            if not pool_op.empty:
                pool = pool_op

        exatos = pool[pool["nome_norm_flex"] == nome_base_flex].copy()

        if len(exatos) == 1:
            status_lista.append("Ativo")
            continue

        if len(exatos) > 1:
            if cpf_base and len(cpf_base) == 11:
                exato_cpf = exatos[exatos["cpf_clean"] == cpf_base].copy()
                if len(exato_cpf) == 1:
                    status_lista.append("Ativo")
                    continue
            status_lista.append("Inativo")
            continue

        if cpf_base and len(cpf_base) == 11:
            cpf_hit = pool[pool["cpf_clean"] == cpf_base].copy()
            if len(cpf_hit) == 1:
                status_lista.append("Ativo")
                continue

        status_lista.append("Inativo")

    base["Status Colaborador"] = status_lista
    return base

# =========================
# ETAPAS
# =========================
ETAPAS = [
    {"chave": "NPS_1_SEMANA", "titulo": "NPS 1ª SEMANA", "tipo": "NPS", "campo_selecao": "Selecione a semana da avaliação:", "valor_selecao": "Primeira semana junto ao padrinho.", "prazo_min_dias": 11, "prazo_max_dias": 14},
    {"chave": "NPS_ULTIMA", "titulo": "NPS ÚLTIMA SEMANA", "tipo": "NPS", "campo_selecao": "Selecione a semana da avaliação:", "valor_selecao": "Última semana junto ao padrinho.", "prazo_min_dias": 20, "prazo_max_dias": 32},
    {"chave": "BP_2_SEMANA", "titulo": "BATE-PAPO — 2ª SEMANA", "tipo": "BP", "campo_selecao": "Selecione a semana do bate papo:", "valor_selecao": "Segunda Semana", "prazo_min_dias": 11, "prazo_max_dias": 14},
    {"chave": "BP_3_SEMANA", "titulo": "BATE-PAPO — 3ª SEMANA", "tipo": "BP", "campo_selecao": "Selecione a semana do bate papo:", "valor_selecao": "Terceira Semana", "prazo_min_dias": 20, "prazo_max_dias": 22},
    {"chave": "BP_ULTIMA", "titulo": "BATE-PAPO — ÚLTIMA SEMANA", "tipo": "BP", "campo_selecao": "Selecione a semana do bate papo:", "valor_selecao": "Última Semana", "prazo_min_dias": 28, "prazo_max_dias": 32},
]

def status_prazo(data_realizacao, prazo_min, prazo_max, hoje):
    if pd.isna(data_realizacao):
        return "Não realizado - Atenção" if hoje <= prazo_max else "Não realizado - Fora do prazo"
    if data_realizacao < prazo_min:
        return "Realizado antes do prazo"
    if data_realizacao <= prazo_max:
        return "Realizado no prazo"
    return "Realizado fora do prazo"

def calcular_dias(status, data_realizacao, prazo_min, prazo_max, hoje):
    if pd.isna(prazo_max):
        return pd.NA
    if pd.isna(data_realizacao):
        return (prazo_max.normalize() - hoje.normalize()).days
    if status == "Realizado fora do prazo":
        return (prazo_max.normalize() - pd.Timestamp(data_realizacao).normalize()).days
    if status == "Realizado antes do prazo":
        if pd.isna(prazo_min):
            return pd.NA
        return (prazo_min.normalize() - pd.Timestamp(data_realizacao).normalize()).days
    if status == "Realizado no prazo":
        return (prazo_max.normalize() - pd.Timestamp(data_realizacao).normalize()).days
    return (prazo_max.normalize() - hoje.normalize()).days

def montar_farol_por_etapa(base_oper, df_nps, df_bp, hoje):
    base = base_oper.copy()
    if "Data_dt" not in base.columns:
        base["Data_dt"] = pd.to_datetime(base["Data"], errors="coerce", dayfirst=True)

    farois = {}
    for etapa in ETAPAS:
        tmp = base[["contrato_id", "Colaborador", "Operação", "Cargo", "Data_dt", "Status Colaborador"]].copy()
        tmp["Etapa"] = etapa["titulo"]
        tmp["Prazo Mín"] = tmp["Data_dt"] + pd.to_timedelta(etapa["prazo_min_dias"], unit="D")
        tmp["Prazo Máx"] = tmp["Data_dt"] + pd.to_timedelta(etapa["prazo_max_dias"], unit="D")

        form = df_nps if etapa["tipo"] == "NPS" else df_bp
        campo = etapa["campo_selecao"]

        if campo not in form.columns:
            tmp["Data Realização"] = pd.NaT
        else:
            form_et = form[form[campo].astype(str).str.strip().eq(etapa["valor_selecao"])].copy()
            col_realizacao = "DataHora Resposta" if "DataHora Resposta" in form_et.columns else "Data Cadastro"
            form_et[col_realizacao] = pd.to_datetime(form_et[col_realizacao], errors="coerce", dayfirst=True)
            real = form_et.dropna(subset=[col_realizacao]).groupby("contrato_id", as_index=False)[col_realizacao].min().rename(columns={col_realizacao: "Data Realização"})
            tmp = tmp.merge(real, on="contrato_id", how="left")
            tmp["Padrinho"] = pd.NA

        tmp["Status"] = tmp.apply(lambda r: status_prazo(r["Data Realização"], r["Prazo Mín"], r["Prazo Máx"], hoje), axis=1)
        tmp["Dias"] = tmp.apply(lambda r: calcular_dias(r["Status"], r["Data Realização"], r["Prazo Mín"], r["Prazo Máx"], hoje), axis=1)

        ordem = pd.CategoricalDtype(categories=["Não realizado - Fora do prazo", "Não realizado - Atenção", "Realizado fora do prazo", "Realizado no prazo", "Realizado antes do prazo"], ordered=True)
        tmp["Status"] = tmp["Status"].astype(ordem)
        farois[etapa["chave"]] = tmp.sort_values(["Status", "Dias"], ascending=[True, True])

    return farois

def render_farol(df_farol: pd.DataFrame, titulo: str, key_prefix: str):
    if df_farol.empty:
        st.info("Sem dados para os filtros selecionados.")
        return

    df_farol = df_farol.copy()
    df_farol["Operação"] = df_farol["Operação"].fillna("SEM OPERAÇÃO").astype(str).str.strip()

    st.markdown(f'<div class="card"><h3 style="margin:0; text-align:center;">{titulo}</h3></div>', unsafe_allow_html=True)

    total = len(df_farol)
    pend_fora = int((df_farol["Status"] == "Não realizado - Fora do prazo").sum())
    mask_atenc_7 = ((df_farol["Status"] == "Não realizado - Atenção") & (pd.to_numeric(df_farol["Dias"], errors="coerce") <= 7))
    pend_atenc_7 = int(mask_atenc_7.sum())
    realizados = int(df_farol["Status"].isin(["Realizado no prazo", "Realizado fora do prazo", "Realizado antes do prazo"]).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{total:,}".replace(",", "."))
    c2.metric("🔴 Pendentes em atraso", f"{pend_fora:,}".replace(",", "."))
    c3.metric("🟡 No prazo vencendo em até 7 dias", f"{pend_atenc_7:,}".replace(",", "."))
    c4.metric("Realizados", f"{realizados:,}".replace(",", "."))

    st.markdown("<hr/>", unsafe_allow_html=True)

    g = df_farol.assign(pend_fora=(df_farol["Status"] == "Não realizado - Fora do prazo")).groupby("Operação", as_index=False).agg(total=("Colaborador", "count"), pend_fora=("pend_fora", "sum"))
    g["Aderência %"] = ((g["total"] - g["pend_fora"]) / g["total"]).fillna(0) * 100
    g["Faixa"] = g["Aderência %"].apply(classificar_faixa_aderencia)
    g = g.sort_values("Aderência %", ascending=False)

    fig = px.bar(g, x="Operação", y="Aderência %", text=g["Aderência %"].round(2).astype(str) + "%", color="Faixa", color_discrete_map={">= 90%": "#2e7d32", "80% a 89%": "#f0d36b", "< 80%": "#c62828"})
    fig.update_traces(textposition="outside", width=0.48)
    fig.update_layout(height=380, template="plotly_dark", yaxis=dict(range=[0, 100], visible=False), xaxis_title="", legend_title="Faixa")
    st.plotly_chart(fig, width="stretch", key=f"chart_{key_prefix}")

    tabela = df_farol[["Operação", "Colaborador", "Cargo", "Data_dt", "Etapa", "Prazo Mín", "Prazo Máx", "Data Realização", "Dias", "Status"]].rename(columns={"Data_dt": "Data Admissão"})
    tabela = formatar_datas_para_tabela(tabela)
    st.dataframe(style_table(tabela), width="stretch", height=430, key=f"df_{key_prefix}")

    excel_bytes = to_excel_bytes(tabela, sheet_name="farol")
    st.download_button("⬇️ Baixar Excel", data=excel_bytes, file_name=f"farol_padrinhos_{key_prefix.lower()}.xlsx", key=f"download_{key_prefix}", width="stretch")

# =========================
# PATHS E CARREGAMENTO
# =========================
DATA_DIR = Path(__file__).parent.parent / "data"
ARQ_ADMITIDOS = DATA_DIR / "Admitidos.xlsx"
ARQ_ATIVOS = DATA_DIR / "Base colaboradores ativos.xlsx"
ARQ_NPS = DATA_DIR / "NPS Mentor.xlsx"
ARQ_BATEPAPO = DATA_DIR / "Bate papo mentor.xlsx"

try:
    admitidos = carregar_excel_primeira_aba(ARQ_ADMITIDOS)
    if OPERACAO_USUARIO != "Todas":
        admitidos = admitidos[admitidos["Operação"] == OPERACAO_USUARIO].copy()
    
    base_ativos = carregar_excel_primeira_aba(ARQ_ATIVOS)
    if OPERACAO_USUARIO != "Todas" and "Operação" in base_ativos.columns:
        base_ativos = base_ativos[base_ativos["Operação"] == OPERACAO_USUARIO].copy()

    nps = carregar_excel_primeira_aba(ARQ_NPS)
    nps = renomear_colunas_duplicadas(nps)
    if OPERACAO_USUARIO != "Todas" and "Informe a operação que você trabalha" in nps.columns:
        nps = nps[nps["Informe a operação que você trabalha"] == OPERACAO_USUARIO].copy()

    batepapo = carregar_excel_primeira_aba(ARQ_BATEPAPO)
    batepapo = renomear_colunas_duplicadas(batepapo)

except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}")
    st.stop()

# =========================
# PIPELINE PRINCIPAL
# =========================
try:
    base_oper = preparar_base_operacional(admitidos, base_ativos)
    base_oper = classificar_status_colaborador(base_oper, base_ativos)
    
    hoje = pd.Timestamp(datetime.now().date())
    farois = montar_farol_por_etapa(base_oper, nps, batepapo, hoje=hoje)

except Exception as e:
    st.error(f"Erro no pipeline: {e}")
    st.stop()

# =========================
# SIDEBAR - FILTROS
# =========================
st.sidebar.markdown("## 🔎 Filtros")

ops_all = sorted([x for x in base_oper["Operação"].fillna("").astype(str).unique().tolist() if x.strip()])
cargos_all = sorted([x for x in base_oper["Cargo"].fillna("").astype(str).unique().tolist() if x.strip()])
colab_all = sorted([x for x in base_oper["Colaborador"].fillna("").astype(str).unique().tolist() if x.strip()])
status_options = ["Não realizado - Fora do prazo", "Não realizado - Atenção", "Realizado fora do prazo", "Realizado no prazo", "Realizado antes do prazo"]
status_colaborador_options = ["Ativo", "Inativo"]

filtro_ops = st.sidebar.multiselect("Operação", options=ops_all, default=[])
filtro_cargos = st.sidebar.multiselect("Cargo", options=cargos_all, default=[])
filtro_colaborador = st.sidebar.multiselect("Colaborador", options=colab_all, default=[])
filtro_status_colaborador = st.sidebar.multiselect("Status do colaborador", options=status_colaborador_options, default=["Ativo"])
filtro_status = st.sidebar.multiselect("Status da etapa", options=status_options, default=[])

def aplicar_filtros_farol(df_farol: pd.DataFrame) -> pd.DataFrame:
    df = df_farol.copy()
    if filtro_ops:
        df = df[df["Operação"].isin(filtro_ops)]
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
# TABS
# =========================
tabs = st.tabs(["PROCESSO PADRINHOS (GERAL)", "NPS 1ª SEMANA", "NPS ÚLTIMA SEMANA", "BATE-PAPO 2ª SEMANA", "BATE-PAPO 3ª SEMANA", "BATE-PAPO ÚLTIMA SEMANA"])

with tabs[0]:
    df_all = pd.concat([farois[e["chave"]] for e in ETAPAS], ignore_index=True)
    render_farol(aplicar_filtros_farol(df_all), "🚦 Processo Padrinhos: Aderência Geral", key_prefix="GERAL")

with tabs[1]:
    render_farol(aplicar_filtros_farol(farois["NPS_1_SEMANA"]), "NPS 1ª SEMANA", key_prefix="NPS1")

with tabs[2]:
    render_farol(aplicar_filtros_farol(farois["NPS_ULTIMA"]), "NPS ÚLTIMA SEMANA", key_prefix="NPSU")

with tabs[3]:
    render_farol(aplicar_filtros_farol(farois["BP_2_SEMANA"]), "BATE-PAPO — 2ª SEMANA", key_prefix="BP2")

with tabs[4]:
    render_farol(aplicar_filtros_farol(farois["BP_3_SEMANA"]), "BATE-PAPO — 3ª SEMANA", key_prefix="BP3")

with tabs[5]:
    render_farol(aplicar_filtros_farol(farois["BP_ULTIMA"]), "BATE-PAPO — ÚLTIMA SEMANA", key_prefix="BPU")

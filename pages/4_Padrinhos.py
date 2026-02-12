# =========================
# FAROL PADRINHOS (cole este bloco inteiro no seu app)
# Pré-requisito: você já tem base_oper, df_nps e df_bp prontos (do pipeline de vínculo)
# =========================

import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

# -------------------------
# Estilo (dark + dourado) - opcional
# -------------------------
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

# -------------------------
# Helpers
# -------------------------
def _to_date(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df

def _status_prazo(data_realizacao, prazo_min, prazo_max, hoje):
    if pd.isna(data_realizacao):
        return "Não realizado - Atenção" if hoje <= prazo_max else "Não realizado - Fora do prazo"
    if data_realizacao < prazo_min:
        return "Realizado antes do prazo"
    if data_realizacao <= prazo_max:
        return "Realizado no prazo"
    return "Realizado fora do prazo"

def _dias_para_prazo_max(prazo_max, hoje):
    if pd.isna(prazo_max):
        return pd.NA
    return (prazo_max.normalize() - hoje.normalize()).days

def _cor_status(status):
    if status == "Realizado no prazo":
        return "background-color: #2e7d32; color: white;"
    if status == "Realizado antes do prazo":
        return "background-color: #1b5e20; color: white;"
    if status == "Realizado fora do prazo":
        return "background-color: #ef6c00; color: black;"
    if status == "Não realizado - Atenção":
        return "background-color: #f0d36b; color: black;"
    if status == "Não realizado - Fora do prazo":
        return "background-color: #c62828; color: white;"
    return ""

def _style_farol_table(df):
    if "Status" in df.columns:
        return (df.style
                .applymap(_cor_status, subset=["Status"])
                .set_properties(**{"text-align": "center"}))
    return df

# -------------------------
# Etapas e prazos (baseados em Data da Admitidos)
# -------------------------
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
        "titulo": "BATE-PAPO PADRINHO — 2ª SEMANA",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Segunda Semana",
        "prazo_min_dias": 11,
        "prazo_max_dias": 14,
    },
    {
        "chave": "BP_3_SEMANA",
        "titulo": "BATE-PAPO PADRINHO — 3ª SEMANA",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Terceira Semana",
        "prazo_min_dias": 20,
        "prazo_max_dias": 22,
    },
    {
        "chave": "BP_ULTIMA",
        "titulo": "BATE-PAPO PADRINHO — ÚLTIMA SEMANA",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Última Semana",
        "prazo_min_dias": 28,
        "prazo_max_dias": 32,
    },
]

# -------------------------
# Montar farol por etapa
# -------------------------
def montar_farol_por_etapa(base_oper, df_nps, df_bp, hoje=None):
    hoje = hoje or pd.Timestamp(datetime.now().date())

    base = base_oper.copy()
    # garantir colunas
    if "Data_dt" not in base.columns:
        base["Data_dt"] = pd.to_datetime(base["Data"], errors="coerce", dayfirst=True)
    if "Operação" not in base.columns:
        base["Operação"] = ""

    # garantir Data Cadastro nas bases de formulário
    df_nps2 = df_nps.copy()
    df_bp2 = df_bp.copy()
    df_nps2 = _to_date(df_nps2, "Data Cadastro")
    df_bp2 = _to_date(df_bp2, "Data Cadastro")

    farois = {}

    for etapa in ETAPAS:
        tmp = base[["Colaborador", "CPF", "cpf_clean", "Operação", "Cargo", "Data_dt"]].copy()
        tmp["Etapa"] = etapa["titulo"]
        tmp["Prazo Mín"] = tmp["Data_dt"] + pd.to_timedelta(etapa["prazo_min_dias"], unit="D")
        tmp["Prazo Máx"] = tmp["Data_dt"] + pd.to_timedelta(etapa["prazo_max_dias"], unit="D")

        form = df_nps2 if etapa["tipo"] == "NPS" else df_bp2
        campo = etapa["campo_selecao"]
        valor = etapa["valor_selecao"]

        # se por algum motivo não existir, marca tudo como não realizado
        if campo not in form.columns:
            tmp["Data Realização"] = pd.NaT
        else:
            form_et = form[form[campo].astype(str).str.strip().eq(valor)].copy()
            # pega a primeira resposta (data mínima) por CPF
            real = (form_et
                    .dropna(subset=["Data Cadastro"])
                    .groupby("cpf_clean", as_index=False)["Data Cadastro"]
                    .min()
                    .rename(columns={"Data Cadastro": "Data Realização"}))
            tmp = tmp.merge(real, on="cpf_clean", how="left")

        tmp["Status"] = tmp.apply(lambda r: _status_prazo(r["Data Realização"], r["Prazo Mín"], r["Prazo Máx"], hoje), axis=1)
        tmp["Dias p/ Prazo Máx"] = tmp["Prazo Máx"].apply(lambda d: _dias_para_prazo_max(d, hoje))

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

# -------------------------
# Render (card + gráfico + lista)
# -------------------------
def render_farol_etapa(df_farol, titulo):
    st.markdown(
        f'<div class="card">'
        f'<h3 style="margin:0; text-align:center;">{titulo}</h3>'
        f'<div class="small-muted" style="text-align:center;">Aderência por operação + lista de pendências para cobrança</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    total = len(df_farol)
    pend_fora = int((df_farol["Status"] == "Não realizado - Fora do prazo").sum())
    pend_atenc = int((df_farol["Status"] == "Não realizado - Atenção").sum())
    ok = int((df_farol["Status"] == "Realizado no prazo").sum())
    fora_real = int((df_farol["Status"] == "Realizado fora do prazo").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", f"{total:,}".replace(",", "."))
    c2.metric("Pend. fora do prazo", f"{pend_fora:,}".replace(",", "."))
    c3.metric("Pend. atenção", f"{pend_atenc:,}".replace(",", "."))
    c4.metric("Realizado no prazo", f"{ok:,}".replace(",", "."))
    c5.metric("Realizado fora do prazo", f"{fora_real:,}".replace(",", "."))

    st.markdown("<hr/>", unsafe_allow_html=True)

    # gráfico: % realizado no prazo por operação
    g = (df_farol.assign(real_no_prazo=(df_farol["Status"] == "Realizado no prazo"))
                .groupby("Operação", as_index=False)
                .agg(total=("Colaborador", "count"), no_prazo=("real_no_prazo", "sum")))
    g["Aderência %"] = (g["no_prazo"] / g["total"]).fillna(0) * 100
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
    fig.update_traces(textposition="outside", cliponaxis=False, marker_color="#f0d36b")  # dourado
    st.plotly_chart(fig, use_container_width=True)

    # lista pendências
    st.markdown('<div class="card"><h4 style="margin:0; text-align:center;">LISTA — PENDÊNCIAS PARA COBRANÇA</h4></div>',
                unsafe_allow_html=True)

    pend = df_farol[df_farol["Status"].isin(["Não realizado - Fora do prazo", "Não realizado - Atenção"])].copy()
    cols_show = ["Operação", "Colaborador", "CPF", "Cargo", "Data_dt", "Prazo Mín", "Prazo Máx", "Dias p/ Prazo Máx", "Status"]
    pend = pend[cols_show].rename(columns={"Data_dt": "Data Admissão"})

    st.dataframe(_style_farol_table(pend), use_container_width=True, height=340)

# =========================
# EXECUÇÃO DO FAROL
# =========================
st.header("🚦 ADERÊNCIA — PROCESSO PADRINHOS (FAROL)")

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
    render_farol_etapa(df_all, "PROCESSO PADRINHOS — ADERÊNCIA GERAL")

with tabs[1]:
    render_farol_etapa(farois["NPS_1_SEMANA"], "NPS 1ª SEMANA")

with tabs[2]:
    render_farol_etapa(farois["NPS_ULTIMA"], "NPS ÚLTIMA SEMANA")

with tabs[3]:
    render_farol_etapa(farois["BP_2_SEMANA"], "BATE-PAPO PADRINHO — 2ª SEMANA")

with tabs[4]:
    render_farol_etapa(farois["BP_3_SEMANA"], "BATE-PAPO PADRINHO — 3ª SEMANA")

with tabs[5]:
    render_farol_etapa(farois["BP_ULTIMA"], "BATE-PAPO PADRINHO — ÚLTIMA SEMANA")

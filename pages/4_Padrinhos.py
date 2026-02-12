import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

# =========================
# Estilo (dark + dourado) - opcional
# =========================
st.markdown("""
<style>
/* fundo geral */
.stApp { background: #0b0b0b; color: #f5f5f5; }

/* títulos */
h1, h2, h3 { color: #f0d36b !important; }

/* cards */
.card {
  background: #000000;
  border-radius: 18px;
  padding: 16px 18px;
  border: 1px solid #222;
  box-shadow: 0 6px 20px rgba(0,0,0,0.35);
  margin-bottom: 14px;
}

/* subtítulo */
.small-muted { color: #bdbdbd; font-size: 0.9rem; }

/* divisor */
hr { border: none; border-top: 1px solid #222; }
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers Farol
# =========================
def _to_date(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df

def _status_prazo(data_realizacao, prazo_min, prazo_max, hoje):
    """
    Regras de status (padrão "Farol"):
    - Se não realizou:
        - Se hoje <= prazo_max  => Não realizado - Atenção
        - Se hoje >  prazo_max  => Não realizado - Fora do prazo
    - Se realizou:
        - Se realizou < prazo_min => Realizado antes do prazo
        - Se prazo_min <= realizou <= prazo_max => Realizado no prazo
        - Se realizou > prazo_max => Realizado fora do prazo
    """
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
    # pra usar no dataframe style
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
    # tabela com status colorido
    if "Status" in df.columns:
        return (df.style
                .applymap(_cor_status, subset=["Status"])
                .set_properties(**{"text-align": "center"})
                )
    return df


# =========================
# Config Etapas e prazos
# =========================
ETAPAS = [
    {
        "chave": "NPS_1_SEMANA",
        "titulo": "NPS 1ª semana",
        "tipo": "NPS",
        "campo_selecao": "Selecione a semana da avaliação:",
        "valor_selecao": "Primeira semana junto ao padrinho.",
        "prazo_min_dias": 11,
        "prazo_max_dias": 14,
    },
    {
        "chave": "NPS_ULTIMA",
        "titulo": "NPS última semana",
        "tipo": "NPS",
        "campo_selecao": "Selecione a semana da avaliação:",
        "valor_selecao": "Última semana junto ao padrinho.",
        "prazo_min_dias": 20,
        "prazo_max_dias": 32,
    },
    {
        "chave": "BP_2_SEMANA",
        "titulo": "Bate-papo 2ª semana",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Segunda Semana",
        "prazo_min_dias": 11,
        "prazo_max_dias": 14,
    },
    {
        "chave": "BP_3_SEMANA",
        "titulo": "Bate-papo 3ª semana",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Terceira Semana",
        "prazo_min_dias": 20,
        "prazo_max_dias": 22,
    },
    {
        "chave": "BP_ULTIMA",
        "titulo": "Bate-papo última semana",
        "tipo": "BP",
        "campo_selecao": "Selecione a semana do bate papo:",
        "valor_selecao": "Última Semana",
        "prazo_min_dias": 28,
        "prazo_max_dias": 32,
    },
]


# =========================
# Construção do Farol por etapa
# =========================
def montar_farol_por_etapa(base_oper, df_nps, df_bp, hoje=None):
    """
    Retorna dict {chave_etapa: dataframe_farol}
    """
    hoje = hoje or pd.Timestamp(datetime.now().date())

    # garantir datas
    base = base_oper.copy()
    base = _to_date(base, "Data_dt")  # já existe no seu pipeline, mas garante
    if "Data_dt" not in base.columns:
        base["Data_dt"] = pd.to_datetime(base["Data"], errors="coerce", dayfirst=True)

    # Sugestão: usar uma coluna operação (se não existir, cria vazio)
    if "Operação" not in base.columns:
        base["Operação"] = ""

    # datas de cadastro nos formulários
    df_nps = df_nps.copy()
    df_bp = df_bp.copy()
    df_nps = _to_date(df_nps, "Data Cadastro")
    df_bp = _to_date(df_bp, "Data Cadastro")

    farois = {}

    for etapa in ETAPAS:
        # monta prazos na base
        tmp = base[["Colaborador", "CPF", "cpf_clean", "Operação", "Cargo", "Data_dt"]].copy()
        tmp["Etapa"] = etapa["titulo"]
        tmp["Prazo Mín"] = tmp["Data_dt"] + pd.to_timedelta(etapa["prazo_min_dias"], unit="D")
        tmp["Prazo Máx"] = tmp["Data_dt"] + pd.to_timedelta(etapa["prazo_max_dias"], unit="D")

        # pega respostas do form certo (NPS ou BP) filtrando pela semana
        if etapa["tipo"] == "NPS":
            form = df_nps
        else:
            form = df_bp

        campo = etapa["campo_selecao"]
        valor = etapa["valor_selecao"]

        if campo not in form.columns:
            # se ainda não existir a coluna (ou nome diferente), deixa sem match
            tmp["Data Realização"] = pd.NaT
            tmp["Status"] = tmp.apply(lambda r: _status_prazo(pd.NaT, r["Prazo Mín"], r["Prazo Máx"], hoje), axis=1)
            tmp["Dias p/ Prazo Máx"] = tmp["Prazo Máx"].apply(lambda d: _dias_para_prazo_max(d, hoje))
            farois[etapa["chave"]] = tmp
            continue

        form_etapa = form[form[campo].astype(str).str.strip().eq(valor)].copy()

        # escolhe data de realização: a primeira resposta (mínima) por cpf_clean
        if "cpf_clean" not in form_etapa.columns:
            form_etapa["cpf_clean"] = ""

        real = (form_etapa
                .dropna(subset=["Data Cadastro"])
                .groupby("cpf_clean", as_index=False)["Data Cadastro"]
                .min()
                .rename(columns={"Data Cadastro": "Data Realização"}))

        tmp = tmp.merge(real, on="cpf_clean", how="left")

        # status e dias
        tmp["Status"] = tmp.apply(lambda r: _status_prazo(r["Data Realização"], r["Prazo Mín"], r["Prazo Máx"], hoje), axis=1)
        tmp["Dias p/ Prazo Máx"] = tmp["Prazo Máx"].apply(lambda d: _dias_para_prazo_max(d, hoje))

        # ordenar por criticidade (fora do prazo primeiro, depois atenção)
        ordem = pd.CategoricalDtype(
            categories=["Não realizado - Fora do prazo", "Não realizado - Atenção", "Realizado fora do prazo", "Realizado no prazo", "Realizado antes do prazo"],
            ordered=True
        )
        tmp["Status"] = tmp["Status"].astype(ordem)
        tmp = tmp.sort_values(["Status", "Dias p/ Prazo Máx"], ascending=[True, True])

        farois[etapa["chave"]] = tmp

    return farois


# =========================
# Render: um "card" por etapa (gráfico + lista)
# =========================
def render_farol_etapa(df_farol, titulo, key_prefix=""):
    st.markdown(f'<div class="card"><h3 style="margin:0">{titulo}</h3>'
                f'<div class="small-muted">Farol de pendentes e aderência por operação</div></div>', unsafe_allow_html=True)

    # resumo
    total = len(df_farol)
    pend_fora = (df_farol["Status"] == "Não realizado - Fora do prazo").sum()
    pend_atenc = (df_farol["Status"] == "Não realizado - Atenção").sum()
    ok = (df_farol["Status"] == "Realizado no prazo").sum()
    fora_real = (df_farol["Status"] == "Realizado fora do prazo").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", f"{total:,}".replace(",", "."))
    c2.metric("Pend. fora do prazo", f"{pend_fora:,}".replace(",", "."))
    c3.metric("Pend. atenção", f"{pend_atenc:,}".replace(",", "."))
    c4.metric("Realizado no prazo", f"{ok:,}".replace(",", "."))
    c5.metric("Realizado fora do prazo", f"{fora_real:,}".replace(",", "."))

    st.markdown("<hr/>", unsafe_allow_html=True)

    # gráfico: aderência por operação (percentual de "Realizado no prazo")
    # (você pode trocar pra "Realizado no prazo + antes" se quiser)
    g = (df_farol.assign(real_no_prazo=(df_farol["Status"] == "Realizado no prazo"))
                  .groupby("Operação", as_index=False)
                  .agg(total=("Colaborador", "count"),
                       no_prazo=("real_no_prazo", "sum")))
    g["Aderência %"] = (g["no_prazo"] / g["total"]).fillna(0) * 100
    g = g.sort_values("Aderência %", ascending=False)

    fig = px.bar(
        g,
        x="Operação",
        y="Aderência %",
        text=g["Aderência %"].round(2).astype(str) + "%",
    )
    fig.update_layout(
        height=320,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 100]),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)

    st.plotly_chart(fig, use_container_width=True)

    # lista: foco pendências (fora do prazo + atenção)
    st.markdown("<div class='card'><h4 style='margin:0'>Lista para cobrança (pendências)</h4></div>", unsafe_allow_html=True)

    pend = df_farol[df_farol["Status"].isin(["Não realizado - Fora do prazo", "Não realizado - Atenção"])].copy()
    cols_show = ["Operação", "Colaborador", "CPF", "Cargo", "Data_dt", "Prazo Mín", "Prazo Máx", "Dias p/ Prazo Máx", "Status"]
    for c in cols_show:
        if c not in pend.columns:
            pend[c] = pd.NA

    pend = pend[cols_show].rename(columns={"Data_dt": "Data Admissão"})
    st.dataframe(_style_farol_table(pend), use_container_width=True, height=280)


# =========================
# ✅ CHAME ISSO NO SEU APP
# =========================
st.header("🚦 Farol de pendentes de realização")

# hoje (pode virar filtro depois)
hoje = pd.Timestamp(datetime.now().date())
farois = montar_farol_por_etapa(base_oper, df_nps, df_bp, hoje=hoje)

# Render por etapa (como teu Power BI: gráfico + lista em cada seção)
for etapa in ETAPAS:
    df_et = farois[etapa["chave"]]
    render_farol_etapa(df_et, etapa["titulo"], key_prefix=etapa["chave"])
    st.divider()

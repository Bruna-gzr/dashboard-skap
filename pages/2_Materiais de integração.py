import streamlit as st

st.set_page_config(page_title="Materiais de Integração", layout="wide")

st.markdown("""
<style>

.stApp{
    background:#050816;
}

.page-title{
    text-align:center;
    color:white;
    font-size:32px;
    margin-bottom:25px;
}

/* ===== FORÇA COLUNAS TEREM MESMA ALTURA ===== */
div[data-testid="column"]{
    display:flex;
    align-items:stretch;
}

div[data-testid="column"] > div{
    width:100%;
    display:flex;
}

/* ===== CARD ===== */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:linear-gradient(135deg,#2D2D2D 0%,#404040 100%);
    border-radius:20px;
    padding:25px 20px;
    margin:8px 0;
    border:1px solid #555;
    box-shadow:0 10px 30px rgba(0,0,0,0.3);

    height:660px;
    width:100%;
    display:flex;
    flex-direction:column;
    box-sizing:border-box;
}

/* ===== TOPO DA UNIDADE (PADRONIZA ALTURA) ===== */
.unidade-header{
    height:240px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:flex-start;
}

/* área fixa da imagem */
div[data-testid="stImage"]{
    height:160px;
    display:flex;
    align-items:center;
    justify-content:center;
}

div[data-testid="stImage"] img{
    object-fit:contain;
}

/* título fixo */
.unidade-titulo{
    min-height:40px;
    display:flex;
    align-items:center;
    justify-content:center;
    color:white;
    font-size:26px;
    font-weight:700;
    text-transform:uppercase;
    margin-bottom:10px;
}

/* colunas */
.titulo-coluna{
    text-align:center;
    color:#CCCCCC;
    font-weight:bold;
    margin-bottom:10px;
}

/* botões */
.stButton button{
    width:100%;
    background:#3A3A3A;
    color:white;
    border:1px solid #555;
    border-radius:8px;
    padding:8px 12px;
    margin:3px 0;
    text-align:left;
}

.stButton button:hover{
    background:#4A4A4A;
}

.logo-fallback{
    background:white;
    border-radius:50%;
    width:120px;
    height:120px;
    display:flex;
    align-items:center;
    justify-content:center;
}

.logo-fallback-grande{
    background:white;
    border-radius:50%;
    width:160px;
    height:160px;
    display:flex;
    align-items:center;
    justify-content:center;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-title'>🧠 Materiais de Integração</h1>", unsafe_allow_html=True)

UNIDADES = {
    "Cascavel":{"logo":"logos/Cascavel.png","coluna1":{"titulo":"🚛 DISTRIBUIÇÃO","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}},
    "Diadema":{"logo":"logos/Diadema.png","coluna1":{"titulo":"🚛 DISTRIBUIÇÃO","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}},
    "Fco Beltrao":{"logo":"logos/Fco Beltrao.png","coluna1":{"titulo":"🚛 DISTRIBUIÇÃO","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}},
    "Foz do Iguacu":{"logo":"logos/Foz do Iguacu.png","coluna1":{"titulo":"🚛 DISTRIBUIÇÃO","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}},
    "Litoral":{"logo":"logos/Litoral.png","coluna1":{"titulo":"🚛 DISTRIBUIÇÃO","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}},
    "Londrina":{"logo":"logos/Londrina.png","coluna1":{"titulo":"🚛 DISTRIBUIÇÃO","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}},
    "Petropolis":{"logo":"logos/Petropolis.png","coluna1":{"titulo":"🚛 DISTRIBUIÇÃO","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}},
    "Ponta Grossa":{"logo":"logos/Ponta Grossa Armazem.png","coluna1":{"titulo":"🚛 EMPURRADA","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}},
    "Sao Cristovao":{"logo":"logos/Sao Cristovao.png","coluna1":{"titulo":"🚛 DISTRIBUIÇÃO","setores":["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]},"coluna2":None},
    "Vidros":{"logo":"logos/Vidros.png","coluna1":None,"coluna2":{"titulo":"👷🏻‍♂️ ARMAZEM","setores":["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]}}
}

ICONES={"GENTE":"👥","SEGURANÇA":"🛡️","ENTREGA":"🚚","FINANCEIRO":"💰","FROTA":"🚛","GESTÃO":"📊","AJUDANTE DE ARMAZEM":"📦","OPERADOR":"🔧"}

UNIDADES_LOGO_GRANDE=["Litoral","Vidros","Londrina","Sao Cristovao"]

def criar_card_unidade(nome_unidade,dados):

    with st.container(border=True):

        logo_grande = nome_unidade in UNIDADES_LOGO_GRANDE
        fallback_class = "logo-fallback-grande" if logo_grande else "logo-fallback"
        logo_width = 160 if logo_grande else 120

        st.markdown("<div class='unidade-header'>",unsafe_allow_html=True)

        try:
            st.image(dados["logo"],width=logo_width)
        except:
            st.markdown(f"<div class='{fallback_class}'><span>🏢</span></div>",unsafe_allow_html=True)

        st.markdown(f"<div class='unidade-titulo'>{nome_unidade}</div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

        colunas_ativas = (dados["coluna1"] is not None) + (dados["coluna2"] is not None)

        if colunas_ativas == 2:
            col1,col2 = st.columns(2)

            with col1:
                st.markdown(f"<div class='titulo-coluna'>{dados['coluna1']['titulo']}</div>",unsafe_allow_html=True)
                for s in dados["coluna1"]["setores"]:
                    st.button(f"{ICONES[s]} {s}",key=f"{nome_unidade}_d_{s}",use_container_width=True)

            with col2:
                st.markdown(f"<div class='titulo-coluna'>{dados['coluna2']['titulo']}</div>",unsafe_allow_html=True)
                for s in dados["coluna2"]["setores"]:
                    st.button(f"{ICONES[s]} {s}",key=f"{nome_unidade}_a_{s}",use_container_width=True)

        elif colunas_ativas == 1:
            col = st.columns([1,2,1])[1]

            with col:
                base = dados["coluna1"] if dados["coluna1"] else dados["coluna2"]
                st.markdown(f"<div class='titulo-coluna'>{base['titulo']}</div>",unsafe_allow_html=True)
                for s in base["setores"]:
                    st.button(f"{ICONES[s]} {s}",key=f"{nome_unidade}_u_{s}",use_container_width=True)

lista=list(UNIDADES.items())

for i in range(0,len(lista),2):
    c1,c2 = st.columns(2,gap="large")

    with c1:
        nome,dados = lista[i]
        criar_card_unidade(nome,dados)

    if i+1 < len(lista):
        with c2:
            nome,dados = lista[i+1]
            criar_card_unidade(nome,dados)

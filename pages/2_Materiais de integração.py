import streamlit as st

st.set_page_config(page_title="Materiais de Integração", layout="wide")

# ================= CSS =================

st.markdown("""
<style>

.stApp{
    background:#050816;
}

.unidade-card{
    background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
    border-radius:20px;
    padding:25px;
    margin-bottom:25px;
    border:1px solid #555;
}

.nome-unidade{
    text-align:center;
    color:white;
    font-size:20px;
    font-weight:700;
    margin-top:10px;
    margin-bottom:20px;
}

.titulo-coluna{
    color:#ccc;
    font-weight:600;
    font-size:14px;
    margin-bottom:6px;
}

.stButton button{
    width:100%;
    background:#3A3A3A;
    color:white;
    border:1px solid #555;
    border-radius:8px;
    margin:4px 0;
}

.stButton button:hover{
    background:#4A4A4A;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:white'>🧠 Materiais de Integração</h1>", unsafe_allow_html=True)

# ================= DADOS =================

UNIDADES = {
    "Cascavel": "logos/Cascavel.png",
    "Diadema": "logos/Diadema.png",
    "Fco Beltrão": "logos/Fco Beltrao.png",
    "Foz do Iguaçu": "logos/Foz do Iguacu.png",
    "Litoral": "logos/Litoral.png",
    "Londrina": "logos/Londrina.png",
    "Petropolis": "logos/Petropolis.png",
    "Ponta Grossa Armazem": "logos/Ponta Grossa Armazem.png",
    "Ponta Grossa Empurrada": "logos/Ponta Grossa Empurrada.png",
    "São Cristóvão": "logos/Sao Cristovao.png",
    "Vidros": "logos/Vidros.png",
}

SET_DIST = ["GENTE","SEGURANÇA","ENTREGA","FINANCEIRO","FROTA"]
SET_ARM = ["GESTÃO","GENTE","SEGURANÇA","AJUDANTE DE ARMAZEM","OPERADOR"]

ICONES = {
    "GENTE":"👥",
    "SEGURANÇA":"🛡️",
    "ENTREGA":"🚚",
    "FINANCEIRO":"💰",
    "FROTA":"🚛",
    "GESTÃO":"📊",
    "AJUDANTE DE ARMAZEM":"📦",
    "OPERADOR":"🔧"
}

# ================= CARD =================

def card(nome, logo):

    st.markdown("<div class='unidade-card'>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns([1,2,1])

    with c2:
        try:
            st.image(logo, width=120)
        except:
            st.write("")

        st.markdown(f"<div class='nome-unidade'>{nome}</div>", unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("<div class='titulo-coluna'>🚛 DISTRIBUIÇÃO</div>", unsafe_allow_html=True)

        for s in SET_DIST:
            st.button(f"{ICONES[s]} {s}", key=f"{nome}{s}d")

    with col2:
        st.markdown("<div class='titulo-coluna'>👷 ARMAZEM</div>", unsafe_allow_html=True)

        for s in SET_ARM:
            st.button(f"{ICONES[s]} {s}", key=f"{nome}{s}a")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= GRID =================

lista = list(UNIDADES.items())

for i in range(0,len(lista),2):

    col1,col2 = st.columns(2,gap="large")

    with col1:
        nome,logo = lista[i]
        card(nome,logo)

    if i+1 < len(lista):
        with col2:
            nome,logo = lista[i+1]
            card(nome,logo)

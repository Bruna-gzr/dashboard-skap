import streamlit as st

# ================= CONFIG =================
st.set_page_config(page_title="Materiais de Integração", layout="wide")

# ================= CSS =================
st.markdown("""
<style>

.stApp{
    background:#050816;
}

.titulo-unidade{
    text-align:center;
    color:white;
    font-size:20px;
    font-weight:700;
    margin-top:8px;
    margin-bottom:20px;
}

.titulo-coluna{
    color:#CCCCCC;
    font-weight:bold;
    font-size:14px;
    margin-bottom:6px;
}

div[data-testid="stVerticalBlockBorderWrapper"]{
    background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
    border-radius:20px;
    padding:20px;
    border:1px solid #555;
    box-shadow:0 10px 30px rgba(0,0,0,0.3);
}

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

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:white'>🧠 Materiais de Integração</h1>", unsafe_allow_html=True)

# ================= DADOS =================
UNIDADES = {
    "Cascavel":{"logo":"logos/Cascavel.png"},
    "Diadema":{"logo":"logos/Diadema.png"},
    "Fco Beltrao":{"logo":"logos/Fco Beltrao.png"},
    "Foz do Iguacu":{"logo":"logos/Foz do Iguacu.png"},
    "Litoral":{"logo":"logos/Litoral.png"},
    "Londrina":{"logo":"logos/Londrina.png"},
    "Petropolis":{"logo":"logos/Petropolis.png"},
    "Ponta Grossa Armazem":{"logo":"logos/Ponta Grossa Armazem.png"},
    "Ponta Grossa Empurrada":{"logo":"logos/Ponta Grossa Empurrada.png"},
    "Sao Cristovao":{"logo":"logos/Sao Cristovao.png"},
    "Vidros":{"logo":"logos/Vidros.png"},
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

# ================= FUNÇÃO CARD =================
def criar_card_unidade(nome_unidade, dados):

    with st.container(border=True):

        # ===== LOGO + TITULO CENTRALIZADOS =====
        esp1, centro, esp2 = st.columns([1,2,1])

        with centro:

            cA, cB, cC = st.columns([1,2,1])

            with cB:
                try:
                    st.image(dados["logo"], width=120)
                except:
                    st.write("")

                st.markdown(
                    f"<div class='titulo-unidade'>{nome_unidade}</div>",
                    unsafe_allow_html=True
                )

        # ===== COLUNAS SETORES =====
        col1,col2 = st.columns(2)

        with col1:
            st.markdown("<div class='titulo-coluna'>🚛 DISTRIBUIÇÃO</div>", unsafe_allow_html=True)

            for s in SET_DIST:
                st.button(f"{ICONES[s]} {s}", key=f"{nome_unidade}_{s}_d")

        with col2:
            st.markdown("<div class='titulo-coluna'>👷🏻‍♂️ ARMAZEM</div>", unsafe_allow_html=True)

            for s in SET_ARM:
                st.button(f"{ICONES[s]} {s}", key=f"{nome_unidade}_{s}_a")

# ================= GRID =================
lista = list(UNIDADES.items())

for i in range(0,len(lista),2):

    col1,col2 = st.columns(2,gap="large")

    with col1:
        nome,dados = lista[i]
        criar_card_unidade(nome,dados)

    if i+1 < len(lista):
        with col2:
            nome,dados = lista[i+1]
            criar_card_unidade(nome,dados)

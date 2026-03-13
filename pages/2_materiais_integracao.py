import streamlit as st

st.set_page_config(page_title="Materiais de Integração", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #050816;
    }
    .page-title {
        text-align: center;
        color: white;
        font-size: 32px;
    }
    .unidade-titulo {
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: 700;
        text-transform: uppercase;
        margin: 20px 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #2D2D2D, #404040);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid #555;
        min-height: 600px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-title'>🧠 Materiais de Integração</h1>", unsafe_allow_html=True)

UNIDADES = {
    "Cascavel": "logos/Cascavel.png",
    "Diadema": "logos/Diadema.png",
    "Fco Beltrao": "logos/Fco Beltrao.png",
    "Foz do Iguacu": "logos/Foz do Iguacu.png",
    "Litoral": "logos/Litoral.png",
    "Londrina": "logos/Londrina.png",
    "Petropolis": "logos/Petropolis.png",
    "Ponta Grossa": "logos/Ponta Grossa Armazem.png",
    "Sao Cristovao": "logos/Sao Cristovao.png",
    "Vidros": "logos/Vidros.png"
}

ICONES = {
    "GENTE": "👥",
    "SEGURANÇA": "🛡️",
    "ENTREGA": "🚚",
    "FINANCEIRO": "💰",
    "FROTA": "🚛",
    "GESTÃO": "📊",
    "AJUDANTE": "📦",
    "OPERADOR": "🔧"
}

LOGO_GRANDE = ["Litoral", "Vidros", "Londrina", "Sao Cristovao"]

def criar_card(nome):
    with st.container(border=True):
        logo_width = 160 if nome in LOGO_GRANDE else 120

        try:
            st.image(UNIDADES[nome], width=logo_width)
        except Exception:
            st.markdown(
                f"<div style='width:{logo_width}px;height:{logo_width}px;background:white;border-radius:50%;margin:0 auto;display:flex;align-items:center;justify-content:center'><span style='font-size:50px'>🏢</span></div>",
                unsafe_allow_html=True
            )

        st.markdown(f"<div class='unidade-titulo'>{nome}</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div style='color:#CCC;text-align:center;font-weight:bold'>🚛 DISTRIBUIÇÃO</div>", unsafe_allow_html=True)
            for s in ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]:
                if st.button(f"{ICONES[s]} {s}", key=f"{nome}_dist_{s}", use_container_width=True):
                    st.info(f"Link {s}")

        with col2:
            st.markdown("<div style='color:#CCC;text-align:center;font-weight:bold'>👷 ARMAZEM</div>", unsafe_allow_html=True)
            for s in ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE", "OPERADOR"]:
                if st.button(f"{ICONES[s]} {s}", key=f"{nome}_arm_{s}", use_container_width=True):
                    st.info(f"Link {s}")

nomes = list(UNIDADES.keys())
for i in range(0, len(nomes), 2):
    cols = st.columns(2)
    with cols[0]:
        criar_card(nomes[i])
    with cols[1]:
        if i + 1 < len(nomes):
            criar_card(nomes[i + 1])

import streamlit as st

# Configuração da página
st.set_page_config(page_title="Materiais de Integração", layout="wide")

# CSS apenas para as cores
st.markdown("""
<style>
    .unidade-card {
        background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
        border-radius: 20px;
        padding: 25px 20px;
        margin: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid #555555;
    }
    
    .stButton button {
        width: 100%;
        background: #3A3A3A;
        color: #FFFFFF;
        border: 1px solid #555555;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 14px;
        text-align: left;
    }
    
    .stButton button:hover {
        background: #4A4A4A;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown("<h1 style='text-align: center;'>🧠 Materiais de Integração</h1>", unsafe_allow_html=True)

# DADOS DAS UNIDADES
UNIDADES = {
    "Cascavel": {
        "logo": "logos/Cascavel.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Diadema": {
        "logo": "logos/Diadema.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    }
}

# ÍCONES
ICONES = {
    "GENTE": "👥", "SEGURANÇA": "🛡️", "ENTREGA": "🚚",
    "FINANCEIRO": "💰", "FROTA": "🚛", "GESTÃO": "📊",
    "AJUDANTE DE ARMAZEM": "📦", "OPERADOR": "🔧"
}

# FUNÇÃO PARA CRIAR O CARD
def criar_card_unidade(nome_unidade, dados):
    with st.container():
        st.markdown(f'<div class="unidade-card">', unsafe_allow_html=True)
        
        # Logo centralizada
        left, center, right = st.columns([1, 2, 1])
        with center:
            try:
                st.image(dados["logo"], width=120)
            except:
                st.markdown("🏢")
        
        # Nome da unidade
        st.markdown(f"<h3 style='text-align: center; color: white;'>{nome_unidade}</h3>", unsafe_allow_html=True)
        
        # Duas colunas para os setores
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{dados['coluna1']['titulo']}**")
            for setor in dados["coluna1"]["setores"]:
                icone = ICONES.get(setor, "🔗")
                st.button(f"{icone} {setor}", key=f"{nome_unidade}_dist_{setor}")
        
        with col2:
            st.markdown(f"**{dados['coluna2']['titulo']}**")
            for setor in dados["coluna2"]["setores"]:
                icone = ICONES.get(setor, "🔗")
                st.button(f"{icone} {setor}", key=f"{nome_unidade}_arm_{setor}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# PÁGINA PRINCIPAL
unidades_lista = list(UNIDADES.items())

for i in range(0, len(unidades_lista), 2):
    cols = st.columns(2)
    
    with cols[0]:
        if i < len(unidades_lista):
            nome, dados = unidades_lista[i]
            criar_card_unidade(nome, dados)
    
    with cols[1]:
        if i + 1 < len(unidades_lista):
            nome, dados = unidades_lista[i + 1]
            criar_card_unidade(nome, dados)

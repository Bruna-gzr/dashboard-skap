import streamlit as st

# Configuração da página
st.set_page_config(page_title="Central de Unidades", layout="wide")

# CSS personalizado para os cards
st.markdown("""
<style>
    .unidade-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px 15px;
        margin: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        height: fit-content;
    }
    
    .logo-container {
        text-align: center;
        background: white;
        border-radius: 50%;
        width: 100px;
        height: 100px;
        margin: 0 auto 15px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .unidade-nome {
        color: white;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
    }
    
    .stButton button {
        width: 100%;
        background: white;
        color: #333;
        border: none;
        border-radius: 10px;
        padding: 12px 15px;
        font-size: 16px;
        font-weight: 500;
        text-align: left;
        margin: 5px 0;
    }
    
    .stButton button:hover {
        background: #f0f0f0;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DADOS DAS UNIDADES (VERSÃO SIMPLIFICADA)
# ============================================

UNIDADES = {
    "Cascavel": {
        "logo": "logos/Cascavel.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Diadema": {
        "logo": "logos/Diadema.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Fco Beltrão": {
        "logo": "logos/Fco Beltrão.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Foz do Iguaçu": {
        "logo": "logos/Foz do Iguaçu.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Litoral": {
        "logo": "logos/Litoral.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Londrina": {
        "logo": "logos/Londrina.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Petropolis": {
        "logo": "logos/Petropolis.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Ponta Grossa Armazem": {
        "logo": "logos/Ponta Grossa Armazem.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Ponta Grossa Empurrada": {
        "logo": "logos/Ponta Grossa Empurrada.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "São Cristovão": {
        "logo": "logos/São Cristovão.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Vidros": {
        "logo": "logos/Vidros.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    }
}

# ============================================
# ÍCONES (sem caracteres especiais)
# ============================================

ICONES = {
    "Gestão": "📊",
    "Distribuição": "📦",
    "Gente": "👥",
    "Segurança": "🛡️",
    "Frota": "🚛",
    "Armazém": "🏭",
    "Financeiro": "💰"
}

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

st.title("🏢 Central de Unidades")

# Criar linhas com 2 colunas cada
unidades_lista = list(UNIDADES.items())

for i in range(0, len(unidades_lista), 2):
    cols = st.columns(2)
    
    # Primeira coluna
    with cols[0]:
        if i < len(unidades_lista):
            nome, dados = unidades_lista[i]
            
            # Card da unidade
            with st.container():
                # Mostrar logo
                try:
                    st.image(dados["logo"], width=80)
                except:
                    st.write("🏢")
                
                # Nome da unidade
                st.subheader(nome)
                
                # Botões dos setores
                for setor in dados["setores"]:
                    icone = ICONES.get(setor, "🔗")
                    if st.button(f"{icone} {setor}", key=f"{nome}_{setor}"):
                        st.info(f"Link para {setor} será adicionado depois")
    
    # Segunda coluna
    with cols[1]:
        if i + 1 < len(unidades_lista):
            nome, dados = unidades_lista[i + 1]
            
            # Card da unidade
            with st.container():
                # Mostrar logo
                try:
                    st.image(dados["logo"], width=80)
                except:
                    st.write("🏢")
                
                # Nome da unidade
                st.subheader(nome)
                
                # Botões dos setores
                for setor in dados["setores"]:
                    icone = ICONES.get(setor, "🔗")
                    if st.button(f"{icone} {setor}", key=f"{nome}_{setor}"):
                        st.info(f"Link para {setor} será adicionado depois")

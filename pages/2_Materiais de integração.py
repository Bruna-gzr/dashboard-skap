import streamlit as st

# Configuração da página
st.set_page_config(page_title="Central de Unidades", layout="wide")

# CSS personalizado para os cards
st.markdown("""
<style>
    /* Container do card */
    .unidade-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px 15px;
        margin: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s;
        height: fit-content;
    }
    
    .unidade-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    /* Área da logo */
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
    
    /* Nome da unidade */
    .unidade-nome {
        color: white;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Container dos botões */
    .botoes-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 0 5px;
    }
    
    /* Estilo dos botões */
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
        transition: all 0.3s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stButton button:hover {
        background: #f0f0f0;
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SEUS DADOS - UNIDADES E LOGOS
# ============================================

UNIDADES = {
    "Cascavel": {
        "logo": "logos/Cascavel.png",  # ← ATENÇÃO: Primeira letra maiúscula
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Diadema": {
        "logo": "logos/Diadema.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Fco Beltrão": {
        "logo": "logos/Fco Beltrão.png",  # ← Com espaço e acento
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Foz do Iguaçu": {
        "logo": "logos/Foz do Iguaçu.png",  # ← Com espaços e acento
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
        "logo": "logos/Ponta Grossa Armazem.png",  ← Com espaços
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Ponta Grossa Empurrada": {
        "logo": "logos/Ponta Grossa Empurrada.png",  ← Com espaços
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "São Cristovão": {
        "logo": "logos/São Cristovão.png",  ← Com acento e espaços
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Vidros": {
        "logo": "logos/Vidros.png",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    }
}

# ============================================
# URLs E ÍCONES (VOCÊ VAI PERSONALIZAR DEPOIS)
# ============================================

URLS_SETORES = {
    "Gestão": "#",
    "Distribuição": "#",
    "Gente": "#",
    "Segurança": "#",
    "Frota": "#",
    "Armazém": "#",
    "Financeiro": "#"
}

ICONES_SETORES = {
    "Gestão": "📊",
    "Distribuição": "📦",
    "Gente": "👥",
    "Segurança": "🛡️",
    "Frota": "🚛",
    "Armazém": "🏭",
    "Financeiro": "💰"
}

# ============================================
# FUNÇÃO PARA CRIAR OS CARDS
# ============================================

def criar_card_unidade(nome_unidade, dados):
    """Cria um card para a unidade com logo e botões verticais"""
    
    with st.container():
        st.markdown(f'<div class="unidade-card">', unsafe_allow_html=True)
        
        # Logo
        st.markdown(f'''
            <div class="logo-container">
                <img src="{dados["logo"]}" width="60" height="60" style="border-radius: 50%;">
            </div>
        ''', unsafe_allow_html=True)
        
        # Nome da unidade
        st.markdown(f'<div class="unidade-nome">{nome_unidade}</div>', unsafe_allow_html=True)
        
        # Container dos botões
        st.markdown('<div class="botoes-container">', unsafe_allow_html=True)
        
        # Botões para cada setor
        for setor in dados["setores"]:
            icone = ICONES_SETORES.get(setor, "🔗")
            url = URLS_SETORES.get(setor, "#")
            
            # Criar botão com key única
            if st.button(f"{icone} {setor}", key=f"{nome_unidade}_{setor}", use_container_width=True):
                js = f"window.open('{url}', '_blank')"
                st.markdown(f'<script>{js}</script>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PÁGINA PRINCIPAL
# ============================================

st.title("🏢 Central de Unidades")
st.markdown("---")

# Organizar unidades em linhas de 2 cards
unidades_lista = list(UNIDADES.items())

for i in range(0, len(unidades_lista), 2):
    # Criar linha com 2 colunas
    cols = st.columns(2, gap="large")
    
    # Primeira unidade da linha
    with cols[0]:
        if i < len(unidades_lista):
            nome, dados = unidades_lista[i]
            criar_card_unidade(nome, dados)
    
    # Segunda unidade da linha
    with cols[1]:
        if i + 1 < len(unidades_lista):
            nome, dados = unidades_lista[i + 1]
            criar_card_unidade(nome, dados)
    
    # Espaço entre linhas
    st.markdown("<br>", unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.caption("🔄 Clique nos botões para acessar os materiais de cada setor")

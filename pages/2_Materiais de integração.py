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
    
    /* Ícones nos botões */
    .botao-icone {
        margin-right: 10px;
        font-size: 18px;
    }
    
    /* Ajuste para mobile */
    @media (max-width: 768px) {
        .unidade-card {
            margin: 5px;
            padding: 15px 10px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Dados das 10 unidades com suas logos e setores
UNIDADES = {
    "Unidade 1 - Centro": {
        "logo": "https://via.placeholder.com/80/ffffff/667eea?text=U1",  # Placeholder com cor
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota", "Armazém", "Financeiro"]
    },
    "Unidade 2 - Norte": {
        "logo": "https://via.placeholder.com/80/ffffff/764ba2?text=U2",
        "setores": ["Gestão", "Distribuição", "Gente", "Segurança", "Frota"]
    },
    "Unidade 3 - Sul": {
        "logo": "https://via.placeholder.com/80/ffffff/667eea?text=U3",
        "setores": ["Gestão", "Armazém", "Gente", "Segurança", "Operador"]
    },
    "Unidade 4 - Leste": {
        "logo": "https://via.placeholder.com/80/ffffff/764ba2?text=U4",
        "setores": ["Gestão", "Distribuição", "Financeiro", "Gente"]
    },
    "Unidade 5 - Oeste": {
        "logo": "https://via.placeholder.com/80/ffffff/667eea?text=U5",
        "setores": ["Gestão", "Frota", "Manutenção", "Segurança", "Armazém"]
    },
    "Unidade 6 - Industrial": {
        "logo": "https://via.placeholder.com/80/ffffff/764ba2?text=U6",
        "setores": ["Gestão", "Produção", "Qualidade", "Segurança", "Manutenção"]
    },
    "Unidade 7 - Comercial": {
        "logo": "https://via.placeholder.com/80/ffffff/667eea?text=U7",
        "setores": ["Vendas", "Marketing", "Financeiro", "Gente", "Jurídico"]
    },
    "Unidade 8 - Logística": {
        "logo": "https://via.placeholder.com/80/ffffff/764ba2?text=U8",
        "setores": ["Distribuição", "Armazém", "Frota", "Roteirização", "Expedição"]
    },
    "Unidade 9 - Matriz": {
        "logo": "https://via.placeholder.com/80/ffffff/667eea?text=U9",
        "setores": ["Diretoria", "Gestão", "Financeiro", "Jurídico", "Gente", "TI"]
    },
    "Unidade 10 - Filial": {
        "logo": "https://via.placeholder.com/80/ffffff/764ba2?text=U10",
        "setores": ["Gestão", "Operações", "Gente", "Segurança", "Manutenção"]
    }
}

# Dicionário de URLs para cada setor (exemplo - você deve preencher com suas URLs reais)
URLS_SETORES = {
    "Gestão": "https://docs.google.com/spreadsheets/d/gestao",
    "Distribuição": "https://docs.google.com/spreadsheets/d/distribuicao",
    "Gente": "https://docs.google.com/forms/d/gente",
    "Segurança": "https://canva.com/seguranca",
    "Frota": "https://docs.google.com/spreadsheets/d/frota",
    "Armazém": "https://canva.com/armazem",
    "Financeiro": "https://docs.google.com/spreadsheets/d/financeiro",
    "Operador": "https://docs.google.com/spreadsheets/d/operador",
    "Manutenção": "https://docs.google.com/spreadsheets/d/manutencao",
    "Produção": "https://docs.google.com/spreadsheets/d/producao",
    "Qualidade": "https://docs.google.com/forms/d/qualidade",
    "Vendas": "https://docs.google.com/spreadsheets/d/vendas",
    "Marketing": "https://canva.com/marketing",
    "Jurídico": "https://docs.google.com/document/d/juridico",
    "Roteirização": "https://maps.google.com/roteiros",
    "Expedição": "https://docs.google.com/spreadsheets/d/expedicao",
    "Diretoria": "https://docs.google.com/presentation/d/diretoria",
    "TI": "https://docs.google.com/forms/d/ti",
    "Operações": "https://docs.google.com/spreadsheets/d/operacoes"
}

# Mapeamento de ícones para cada setor
ICONES_SETORES = {
    "Gestão": "📊",
    "Distribuição": "📦",
    "Gente": "👥",
    "Segurança": "🛡️",
    "Frota": "🚛",
    "Armazém": "🏭",
    "Financeiro": "💰",
    "Operador": "🔧",
    "Manutenção": "🔨",
    "Produção": "⚙️",
    "Qualidade": "✅",
    "Vendas": "📈",
    "Marketing": "🎯",
    "Jurídico": "⚖️",
    "Roteirização": "🗺️",
    "Expedição": "📬",
    "Diretoria": "👔",
    "TI": "💻",
    "Operações": "🔄"
}

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

# Título da página
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

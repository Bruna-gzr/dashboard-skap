# 1_Painel de Acessos RH.py
import streamlit as st

# Configuração da página
st.set_page_config(page_title="Materiais de Gestão RH", layout="wide")

# CSS (mesmo estilo da página original)
st.markdown("""
<style>
    .stApp {
        background-color: #050816;
    }

    .page-title {
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }

    .unidade-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    
    .unidade-titulo {
        text-align: center;
        color: white;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        padding: 0;
        margin-bottom: 25px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
        border-radius: 20px;
        padding: 25px 20px 25px 20px;
        margin: 8px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid #555555;
        height: fit-content;
    }

    div[data-testid="stImage"] {
        text-align: center;
    }

    div[data-testid="stImage"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    .logo-fallback {
        background: white;
        border-radius: 50%;
        width: 120px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px auto;
    }
    
    .logo-fallback-grande {
        background: white;
        border-radius: 50%;
        width: 160px !important;
        height: 160px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px auto;
    }
    
    .logo-fallback span, .logo-fallback-grande span {
        font-size: 60px;
    }

    .link-botao {
        display: block;
        width: 100%;
        background: #3A3A3A;
        color: #FFFFFF !important;
        border: 1px solid #555555;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
        font-weight: 500;
        text-align: left;
        text-decoration: none !important;
        margin: 6px 0;
        box-sizing: border-box;
    }

    .link-botao:hover {
        background: #4A4A4A;
        color: #FFFFFF !important;
        border: 1px solid #777777;
        text-decoration: none !important;
    }

    .link-botao-vazio {
        display: block;
        width: 100%;
        background: #2A2A2A;
        color: #999999 !important;
        border: 1px dashed #555555;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
        font-weight: 500;
        text-align: left;
        text-decoration: none !important;
        margin: 6px 0;
        box-sizing: border-box;
        cursor: default;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown("<h1 class='page-title'>📊 MATERIAIS DE GESTÃO RH</h1>", unsafe_allow_html=True)

# ============================================
# DADOS DAS UNIDADES
# ============================================

UNIDADES = {
    "Cascavel": {
        "logo": "logos/Cascavel.png",
    },
    "Diadema": {
        "logo": "logos/Diadema.png",
    },
    "Fco Beltrao": {
        "logo": "logos/Fco Beltrao.png",
    },
    "Foz do Iguacu": {
        "logo": "logos/Foz do Iguacu.png",
    },
    "Litoral": {
        "logo": "logos/Litoral.png",
    },
    "Londrina": {
        "logo": "logos/Londrina.png",
    },
    "Petropolis": {
        "logo": "logos/Petropolis.png",
    },
    "Ponta Grossa": {
        "logo": "logos/Ponta Grossa Armazem.png",
    },
    "Sao Cristovao": {
        "logo": "logos/Sao Cristovao.png",
    },
    "Vidros": {
        "logo": "logos/Vidros.png",
    }
}

# Ícones para cada botão
ICONES = {
    "Comitê de Gente": "👥",
    "RPS de gente": "📋",
    "Café com Gerente": "☕",
    "Pipeline": "📊",
    "Check List DPO/VPO": "✅",
    "Defesa DPO/VPO": "🛡️"
}

# LISTA DOS BOTÕES NA ORDEM DESEJADA
BOTOES = ["Comitê de Gente", "RPS de gente", "Café com Gerente", "Pipeline", "Check List DPO/VPO", "Defesa DPO/VPO"]

# LINKS PARA CADA UNIDADE E CADA BOTÃO
# Substitua os links pelos links reais de cada unidade
LINKS = {
    "Cascavel": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Diadema": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Fco Beltrao": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Foz do Iguacu": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Litoral": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Londrina": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Petropolis": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Ponta Grossa": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Sao Cristovao": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    },
    "Vidros": {
        "Comitê de Gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "RPS de gente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Café com Gerente": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Pipeline": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Check List DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
        "Defesa DPO/VPO": "https://www.canva.com/design/SEU_LINK_AQUI",
    }
}

UNIDADES_LOGO_GRANDE = ["Litoral", "Vidros"]

def render_link_botao(label, link):
    if link:
        st.markdown(
            f'<a class="link-botao" href="{link}" target="_blank">{label}</a>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="link-botao-vazio">{label}</div>',
            unsafe_allow_html=True
        )

def criar_card_unidade(nome_unidade, dados):
    with st.container(border=True):
        logo_grande = nome_unidade in UNIDADES_LOGO_GRANDE
        fallback_class = "logo-fallback-grande" if logo_grande else "logo-fallback"
        logo_width = 160 if logo_grande else 120
        
        st.markdown('<div class="unidade-header">', unsafe_allow_html=True)
        
        try:
            st.image(dados["logo"], width=logo_width)
        except Exception:
            st.markdown(f"""
            <div class="{fallback_class}">
                <span>🏢</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="unidade-titulo">{nome_unidade}</div>
        </div>
        """, unsafe_allow_html=True)

        # Renderiza os 6 botões em sequência
        for botao in BOTOES:
            icone = ICONES.get(botao, "🔗")
            link = LINKS.get(nome_unidade, {}).get(botao, "")
            render_link_botao(f"{icone} {botao}", link)

unidades_lista = list(UNIDADES.items())

for i in range(0, len(unidades_lista), 2):
    cols = st.columns(2, gap="large")

    with cols[0]:
        if i < len(unidades_lista):
            nome, dados = unidades_lista[i]
            criar_card_unidade(nome, dados)

    with cols[1]:
        if i + 1 < len(unidades_lista):
            nome, dados = unidades_lista[i + 1]
            criar_card_unidade(nome, dados)

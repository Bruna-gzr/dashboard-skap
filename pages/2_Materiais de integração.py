import streamlit as st

# Configuração da página
st.set_page_config(page_title="Materiais de Integração", layout="wide")

# CSS
st.markdown("""
<style>
    .stApp {
        background-color: #050816;
    }

    .page-title {
        text-align: center;
        color: white;
        margin-bottom: 20px;
        font-size: 32px;
    }

    /* Container do cabeçalho da unidade */
    .unidade-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    
    .unidade-logo {
        width: 120px;
        height: 120px;
        object-fit: contain;
        margin-bottom: 15px;
        transition: transform 0.3s ease;  /* Efeito hover suave */
    }
    
    .unidade-logo:hover {
        transform: scale(1.05);  /* Efeito de zoom ao passar o mouse */
    }
    
    /* Classe especial para logos maiores */
    .unidade-logo-grande {
        width: 160px !important;
        height: 160px !important;
        object-fit: contain;
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    
    .unidade-logo-grande:hover {
        transform: scale(1.05);
    }
    
    .unidade-titulo {
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        padding: 0;
        margin-bottom: 25px;
        letter-spacing: 0.5px;
        text-transform: uppercase;  /* CAIXA ALTA */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);  /* Sombra suave */
    }

    .titulo-coluna {
        color: #CCCCCC;
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 16px;
        text-align: center;
        width: 100%;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #555555;  /* Linha decorativa abaixo do título */
        padding-bottom: 8px;
    }

    /* Card com visual parecido com o seu */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
        border-radius: 20px;
        padding: 25px 20px 25px 20px;
        margin: 8px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid #555555;
        height: 100%;
        min-height: 600px;  /* Aumentado ligeiramente */
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        border-color: #777777;
        transform: translateY(-2px);
    }
    
    /* Garante que o conteúdo interno use todo o espaço disponível */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #3A3A3A 0%, #454545 100%);  /* Gradiente suave */
        color: #FFFFFF;
        border: 1px solid #555555;
        border-radius: 10px;  /* Aumentado de 8px para 10px */
        padding: 10px 12px;  /* Aumentado padding vertical */
        font-size: 14px;
        font-weight: 500;
        text-align: left;
        margin: 4px 0;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #4A4A4A 0%, #555555 100%);
        color: #FFFFFF;
        border: 1px solid #888888;
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
    }
    
    .stButton button:active {
        transform: translateY(0);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Fallback para logo */
    .logo-fallback {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F0F0 100%);
        border-radius: 50%;
        width: 120px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px auto;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .logo-fallback:hover {
        transform: scale(1.05);
    }
    
    /* Fallback maior para unidades específicas */
    .logo-fallback-grande {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F0F0 100%);
        border-radius: 50%;
        width: 160px !important;
        height: 160px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px auto;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .logo-fallback-grande:hover {
        transform: scale(1.05);
    }
    
    .logo-fallback span, .logo-fallback-grande span {
        font-size: 70px;
    }
    
    /* Ajuste para as colunas ficarem com altura consistente */
    div[data-testid="column"] {
        height: 100%;
    }
    
    /* Container dos botões para distribuição uniforme */
    .botoes-container {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    
    /* Para cards com apenas uma coluna */
    .coluna-unica {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
    }
    
    /* Barra de rolagem personalizada (opcional) */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2D2D2D;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #555555;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #777777;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown("<h1 class='page-title'>🧠 Materiais de Integração</h1>", unsafe_allow_html=True)

# ============================================
# DADOS DAS UNIDADES
# ============================================

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
    },
    "Fco Beltrao": {
        "logo": "logos/Fco Beltrao.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Foz do Iguacu": {
        "logo": "logos/Foz do Iguacu.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Litoral": {
        "logo": "logos/Litoral.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Londrina": {
        "logo": "logos/Londrina.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Petropolis": {
        "logo": "logos/Petropolis.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Ponta Grossa": {
        "logo": "logos/Ponta Grossa Armazem.png",
        "coluna1": {
            "titulo": "🚛 EMPURRADA",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Sao Cristovao": {
        "logo": "logos/Sao Cristovao.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": None
    },
    "Vidros": {
        "logo": "logos/Vidros.png",
        "coluna1": None,
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    }
}

# Mapeamento de ícones
ICONES = {
    "GENTE": "👥",
    "SEGURANÇA": "🛡️",
    "ENTREGA": "🚚",
    "FINANCEIRO": "💰",
    "FROTA": "🚛",
    "GESTÃO": "📊",
    "AJUDANTE DE ARMAZEM": "📦",
    "OPERADOR": "🔧"
}

# Unidades que terão logo maior (adicionadas Londrina e Sao Cristovao)
UNIDADES_LOGO_GRANDE = ["Litoral", "Vidros", "Londrina", "Sao Cristovao"]

# ============================================
# FUNÇÃO PARA CRIAR O CARD DA UNIDADE
# ============================================

def criar_card_unidade(nome_unidade, dados):
    with st.container(border=True):
        
        # Verifica se a unidade deve ter logo maior
        logo_grande = nome_unidade in UNIDADES_LOGO_GRANDE
        
        # Define classes CSS baseado no tamanho da logo
        logo_class = "unidade-logo-grande" if logo_grande else "unidade-logo"
        fallback_class = "logo-fallback-grande" if logo_grande else "logo-fallback"
        logo_width = 160 if logo_grande else 120
        
        # Cabeçalho com logo e título alinhados
        st.markdown(f"""
        <div class="unidade-header">
        """, unsafe_allow_html=True)
        
        # Tenta carregar a imagem, se não conseguir mostra fallback
        try:
            # Para imagens, usamos o parâmetro width do st.image
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

        # Verifica quantas colunas precisamos criar
        colunas_ativas = 0
        if dados["coluna1"] is not None:
            colunas_ativas += 1
        if dados["coluna2"] is not None:
            colunas_ativas += 1
        
        # Se tiver duas colunas ativas
        if colunas_ativas == 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    f"<div class='titulo-coluna'>{dados['coluna1']['titulo']}</div>",
                    unsafe_allow_html=True
                )
                for setor in dados["coluna1"]["setores"]:
                    icone = ICONES.get(setor, "🔗")
                    if st.button(
                        f"{icone} {setor}",
                        key=f"{nome_unidade}_dist_{setor}",
                        use_container_width=True
                    ):
                        st.info(f"Link para {setor}")

            with col2:
                st.markdown(
                    f"<div class='titulo-coluna'>{dados['coluna2']['titulo']}</div>",
                    unsafe_allow_html=True
                )
                for setor in dados["coluna2"]["setores"]:
                    icone = ICONES.get(setor, "🔗")
                    if st.button(
                        f"{icone} {setor}",
                        key=f"{nome_unidade}_arm_{setor}",
                        use_container_width=True
                    ):
                        st.info(f"Link para {setor}")
        
        # Se tiver apenas uma coluna ativa (Vidros ou Sao Cristovao)
        elif colunas_ativas == 1:
            # Usa coluna centralizada
            col = st.columns([1, 2, 1])[1]
            
            with col:
                if dados["coluna1"] is not None:
                    st.markdown(
                        f"<div class='titulo-coluna'>{dados['coluna1']['titulo']}</div>",
                        unsafe_allow_html=True
                    )
                    for setor in dados["coluna1"]["setores"]:
                        icone = ICONES.get(setor, "🔗")
                        if st.button(
                            f"{icone} {setor}",
                            key=f"{nome_unidade}_dist_{setor}",
                            use_container_width=True
                        ):
                            st.info(f"Link para {setor}")
                
                elif dados["coluna2"] is not None:
                    st.markdown(
                        f"<div class='titulo-coluna'>{dados['coluna2']['titulo']}</div>",
                        unsafe_allow_html=True
                    )
                    for setor in dados["coluna2"]["setores"]:
                        icone = ICONES.get(setor, "🔗")
                        if st.button(
                            f"{icone} {setor}",
                            key=f"{nome_unidade}_arm_{setor}",
                            use_container_width=True
                        ):
                            st.info(f"Link para {setor}")

# ============================================
# PÁGINA PRINCIPAL
# ============================================

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

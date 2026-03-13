import streamlit as st

# Configuração da página
st.set_page_config(page_title="Materiais de Integração", layout="wide")

# CSS - VERSÃO SIMPLIFICADA PARA GARANTIR QUE FUNCIONE
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
    
    /* CLASSE PARA LOGOS NORMAIS */
    .unidade-logo {
        width: 120px !important;
        height: 120px !important;
        object-fit: contain;
        margin-bottom: 15px;
    }
    
    /* CLASSE PARA LOGOS GRANDES */
    .unidade-logo-grande {
        width: 160px !important;
        height: 160px !important;
        object-fit: contain;
        margin-bottom: 15px;
    }
    
    /* TÍTULO DA UNIDADE - CAIXA ALTA */
    .unidade-titulo {
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        padding: 0;
        margin-bottom: 25px;
        text-transform: uppercase !important; /* FORÇA CAIXA ALTA */
    }

    .titulo-coluna {
        color: #CCCCCC;
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 16px;
        text-align: center;
        width: 100%;
    }

    /* Card */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
        border-radius: 20px;
        padding: 25px 20px 25px 20px;
        margin: 8px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid #555555;
        height: 100%;
        min-height: 600px;
    }

    .stButton button {
        width: 100%;
        background: #3A3A3A;
        color: #FFFFFF;
        border: 1px solid #555555;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 14px;
        font-weight: 500;
        text-align: left;
        margin: 3px 0;
    }

    .stButton button:hover {
        background: #4A4A4A;
        border: 1px solid #777777;
    }
    
    /* Fallback para logo */
    .logo-fallback {
        background: white;
        border-radius: 50%;
        width: 120px !important;
        height: 120px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px auto;
    }
    
    /* Fallback maior */
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
        font-size: 70px;
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

# Unidades que terão logo maior - ATUALIZADO
UNIDADES_LOGO_GRANDE = ["Litoral", "Vidros", "Londrina", "Sao Cristovao"]

# ============================================
# FUNÇÃO PARA CRIAR O CARD DA UNIDADE
# ============================================

def criar_card_unidade(nome_unidade, dados):
    with st.container(border=True):
        
        # Verifica se a unidade deve ter logo maior
        logo_grande = nome_unidade in UNIDADES_LOGO_GRANDE
        
        # Define classes CSS baseado no tamanho da logo
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
        
        # Se tiver apenas uma coluna ativa
        elif colunas_ativas == 1:
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

import streamlit as st

# Configuração da página
st.set_page_config(page_title="Materiais de Integração", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    /* Título centralizado */
    .titulo-central {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 2rem;
        color: #FFFFFF;
    }
    
    /* Container do card da unidade - CINZA ESCURO ELEGANTE */
    .unidade-card {
        background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
        border-radius: 20px;
        padding: 25px 20px;
        margin: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid #555555;
    }
    
    /* Logo container */
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
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Nome da unidade */
    .unidade-nome {
        color: white;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 25px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255,255,255,0.2);
    }
    
    /* Títulos das seções */
    .titulo-secao {
        color: #CCCCCC;
        font-size: 18px;
        font-weight: bold;
        margin: 15px 0 10px 0;
        padding-left: 5px;
    }
    
    /* Estilo dos botões */
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
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #4A4A4A;
        transform: translateX(5px);
        color: #FFFFFF;
        border-color: #777777;
    }
</style>
""", unsafe_allow_html=True)

# Título centralizado
st.markdown('<h1 class="titulo-central">🧠 Materiais de Integração</h1>', unsafe_allow_html=True)

# ============================================
# DADOS DAS UNIDADES COM CAMINHOS CORRIGIDOS
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
    "Ponta Grossa Armazem": {
        "logo": "logos/Ponta Grossa Armazem.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Ponta Grossa Empurrada": {
        "logo": "logos/Ponta Grossa Empurrada.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
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
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR"]
        }
    },
    "Vidros": {
        "logo": "logos/Vidros.png",
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

# ============================================
# FUNÇÃO PARA CRIAR O CARD DA UNIDADE
# ============================================

def criar_card_unidade(nome_unidade, dados):
    """Cria um card para a unidade com logo e duas colunas internas"""
    
    with st.container():
        st.markdown(f'<div class="unidade-card">', unsafe_allow_html=True)
        
        # ===== LOGO COM TRATAMENTO DE ERRO =====
        try:
            # Tenta carregar a imagem do caminho especificado
            st.image(dados["logo"], width=80)
        except Exception as e:
            # Se falhar, mostra um placeholder bonito com emoji
            st.markdown(f'''
                <div class="logo-container">
                    <span style="font-size: 40px;">🏢</span>
                </div>
            ''', unsafe_allow_html=True)
        # =======================================
        
        # Nome da unidade
        st.markdown(f'<div class="unidade-nome">{nome_unidade}</div>', unsafe_allow_html=True)
        
        # Criar duas colunas dentro do card
        col_esquerda, col_direita = st.columns(2)
        
        # Coluna Esquerda - DISTRIBUIÇÃO
        with col_esquerda:
            st.markdown(f'<p class="titulo-secao">{dados["coluna1"]["titulo"]}</p>', unsafe_allow_html=True)
            for setor in dados["coluna1"]["setores"]:
                icone = ICONES.get(setor, "🔗")
                if st.button(f"{icone} {setor}", key=f"{nome_unidade}_dist_{setor}", use_container_width=True):
                    st.info(f"Link para {setor} será adicionado depois")
        
        # Coluna Direita - ARMAZEM
        with col_direita:
            st.markdown(f'<p class="titulo-secao">{dados["coluna2"]["titulo"]}</p>', unsafe_allow_html=True)
            for setor in dados["coluna2"]["setores"]:
                icone = ICONES.get(setor, "🔗")
                if st.button(f"{icone} {setor}", key=f"{nome_unidade}_arm_{setor}", use_container_width=True):
                    st.info(f"Link para {setor} será adicionado depois")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PÁGINA PRINCIPAL
# ============================================

# Organizar unidades em linhas de 2 cards
unidades_lista = list(UNIDADES.items())

for i in range(0, len(unidades_lista), 2):
    # Criar linha com 2 colunas para as unidades
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

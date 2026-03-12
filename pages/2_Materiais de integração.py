import streamlit as st

# Configuração da página
st.set_page_config(page_title="Materiais de Integração", layout="wide")

# CSS
st.markdown("""
<style>
    .stApp {
        background-color: #050816;
    }

    h1 {
        color: white;
    }

    /* Card visual */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
        border-radius: 20px;
        padding: 18px 18px 22px 18px;
        border: 1px solid #555555;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .titulo-unidade {
        text-align: center;
        color: white;
        margin: 6px 0 20px 0;
        font-size: 20px;
        font-weight: 700;
    }

    .titulo-coluna {
        color: #CCCCCC;
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 14px;
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
        color: #FFFFFF;
        border: 1px solid #777777;
    }

    /* tira excesso de espaço do image */
    div[data-testid="stImage"] {
        text-align: center;
    }

    div[data-testid="stImage"] img {
        margin: 0 auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown(
    "<h1 style='text-align: center;'>🧠 Materiais de Integração</h1>",
    unsafe_allow_html=True
)

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
    with st.container(border=True):

        # Logo centralizada
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            try:
                st.image(dados["logo"], width=120)
            except Exception:
                st.markdown(
                    """
                    <div style='
                        background: white;
                        border-radius: 50%;
                        width: 120px;
                        height: 120px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto;
                    '>
                        <span style='font-size: 60px;'>🏢</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Nome da unidade centralizado
        st.markdown(
            f"<div class='titulo-unidade'>{nome_unidade}</div>",
            unsafe_allow_html=True
        )

        # Duas colunas para os setores
        col1, col2 = st.columns(2)

        # Coluna 1 - DISTRIBUIÇÃO
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

        # Coluna 2 - ARMAZEM
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

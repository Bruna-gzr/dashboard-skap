import streamlit as st
import base64
from pathlib import Path

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(page_title="Materiais de Integração", layout="wide")

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
    .stApp {
        background-color: #050816;
    }

    .page-title {
        text-align: center;
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 30px;
    }

    .unidade-card {
        background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
        border-radius: 20px;
        padding: 28px 22px;
        margin: 10px 0 18px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.28);
        border: 1px solid #555555;
        min-height: 760px;
    }

    .logo-wrap {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 12px;
    }

    .logo-wrap img {
        max-width: 120px;
        max-height: 120px;
        object-fit: contain;
        display: block;
    }

    .logo-fallback {
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: white;
        color: black;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 56px;
    }

    .unidade-title {
        text-align: center;
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin: 0 0 26px 0;
    }

    .duas-colunas {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }

    .coluna-titulo {
        color: #D7D7D7;
        font-size: 14px;
        font-weight: 700;
        margin: 0 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .setor-link {
        display: block;
        width: 100%;
        text-decoration: none;
        background: #3A3A3A;
        color: #FFFFFF !important;
        border: 1px solid #555555;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
        font-weight: 500;
        text-align: center;
        margin: 8px 0;
        transition: 0.2s ease-in-out;
        box-sizing: border-box;
    }

    .setor-link:hover {
        background: #4A4A4A;
        border-color: #777777;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TÍTULO
# ============================================
st.markdown("<div class='page-title'>🧠 Materiais de Integração</div>", unsafe_allow_html=True)

# ============================================
# FUNÇÕES AUXILIARES
# ============================================
def imagem_para_base64(caminho_imagem: str):
    try:
        path = Path(caminho_imagem)
        if not path.exists():
            return None

        extensao = path.suffix.lower().replace(".", "")
        if extensao == "jpg":
            extensao = "jpeg"

        with open(path, "rb") as f:
            img_bytes = f.read()

        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/{extensao};base64,{img_base64}"
    except Exception:
        return None


def montar_links(setores):
    html = ""
    for setor in setores:
        icone = ICONES.get(setor, "🔗")
        link = LINKS.get(setor, "#")
        html += f'<a class="setor-link" href="{link}" target="_blank">{icone} {setor}</a>'
    return html


def criar_card_unidade(nome_unidade, dados):
    logo_src = imagem_para_base64(dados["logo"])

    if logo_src:
        logo_html = f"""
        <div class="logo-wrap">
            <img src="{logo_src}" alt="{nome_unidade}">
        </div>
        """
    else:
        logo_html = """
        <div class="logo-wrap">
            <div class="logo-fallback">🏢</div>
        </div>
        """

    html = f"""
    <div class="unidade-card">
        {logo_html}
        <div class="unidade-title">{nome_unidade}</div>

        <div class="duas-colunas">
            <div>
                <div class="coluna-titulo">{dados['coluna1']['titulo']}</div>
                {montar_links(dados['coluna1']['setores'])}
            </div>

            <div>
                <div class="coluna-titulo">{dados['coluna2']['titulo']}</div>
                {montar_links(dados['coluna2']['setores'])}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ============================================
# LINKS DOS SETORES
# TROQUE PELOS LINKS REAIS
# ============================================
LINKS = {
    "GENTE": "#",
    "SEGURANÇA": "#",
    "ENTREGA": "#",
    "FINANCEIRO": "#",
    "FROTA": "#",
    "GESTÃO": "#",
    "AJUDANTE DE ARMAZEM": "#",
    "OPERADOR": "#"
}

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

# ============================================
# MAPEAMENTO DE ÍCONES
# ============================================
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
# PÁGINA PRINCIPAL
# ============================================
unidades_lista = list(UNIDADES.items())

for i in range(0, len(unidades_lista), 2):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        if i < len(unidades_lista):
            nome, dados = unidades_lista[i]
            criar_card_unidade(nome, dados)

    with col2:
        if i + 1 < len(unidades_lista):
            nome, dados = unidades_lista[i + 1]
            criar_card_unidade(nome, dados)

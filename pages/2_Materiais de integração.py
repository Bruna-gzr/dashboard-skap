import streamlit as st

# Configuração da página
st.set_page_config(page_title="Central de Links por Unidade", layout="wide")

# CSS personalizado para os cards
st.markdown("""
<style>
    .unidade-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        border: 1px solid #ddd;
    }
    .unidade-titulo {
        color: #0e1117;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }
    .area-botao {
        margin: 5px 0;
        width: 100%;
    }
    .stButton button {
        width: 100%;
        background-color: white;
        color: #0e1117;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #e6e9ef;
        border-color: #9c27b0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Dados completos das unidades e suas áreas
UNIDADES = {
    "Cascavel": {
        "logo": "logos/cascavel.png",  # Caminho da logo
        "areas": {
            "Gestão": {
                "links": {
                    "Distribuição": "https://docs.google.com/spreadsheets/d/1",
                    "Gente": "https://docs.google.com/forms/d/1",
                    "Segurança": "https://canva.com/seguranca1",
                    "Frota": "https://docs.google.com/spreadsheets/d/2",
                    "Armazem": "https://canva.com/armazem1",
                    "Financeiro": "https://docs.google.com/spreadsheets/d/3"
                }
            }
        }
    },
    "Maringá": {
        "logo": "logos/maringa.png",
        "areas": {
            "Gestão": {
                "links": {
                    "Distribuição": "https://docs.google.com/spreadsheets/d/4",
                    "Gente": "https://docs.google.com/forms/d/2",
                    "Segurança": "https://canva.com/seguranca2",
                    "Frota": "https://docs.google.com/spreadsheets/d/5"
                }
            }
        }
    },
    "Londrina": {
        "logo": "logos/londrina.png",
        "areas": {
            "Gestão": {
                "links": {
                    "Distribuição": "https://docs.google.com/spreadsheets/d/6",
                    "Gente": "https://docs.google.com/forms/d/3",
                    "Segurança": "https://canva.com/seguranca3"
                }
            },
            "Armazém": {
                "links": {
                    "Gente": "https://docs.google.com/forms/d/4",
                    "Segurança": "https://canva.com/seguranca4",
                    "Ajudante": "https://docs.google.com/spreadsheets/d/7",
                    "Operador": "https://docs.google.com/spreadsheets/d/8"
                }
            }
        }
    },
    "Foz do Iguaçu": {
        "logo": "logos/foz.png",
        "areas": {
            "Gestão": {
                "links": {
                    "Distribuição": "https://docs.google.com/spreadsheets/d/9",
                    "Gente": "https://docs.google.com/forms/d/5",
                    "Segurança": "https://canva.com/seguranca5"
                }
            },
            "Armazém": {
                "links": {
                    "Gente": "https://docs.google.com/forms/d/6",
                    "Segurança": "https://canva.com/seguranca6",
                    "Operador": "https://docs.google.com/spreadsheets/d/10"
                }
            },
            "Frota": {
                "links": {
                    "Manutenção": "https://docs.google.com/spreadsheets/d/11",
                    "Abastecimento": "https://forms.gle/abastecimento"
                }
            }
        }
    }
}

def criar_card_unidade(nome_unidade, dados_unidade):
    """Cria um card para uma unidade com suas áreas e links"""
    
    with st.container():
        # Card da unidade
        st.markdown(f'<div class="unidade-card">', unsafe_allow_html=True)
        
        # Logo e nome da unidade
        col_logo, col_titulo = st.columns([1, 3])
        
        with col_logo:
            try:
                st.image(dados_unidade["logo"], width=60)
            except:
                # Se não encontrar a logo, mostra um placeholder
                st.markdown("🏢")
        
        with col_titulo:
            st.markdown(f'<p class="unidade-titulo">{nome_unidade}</p>', unsafe_allow_html=True)
        
        st.divider()
        
        # Para cada área dentro da unidade
        for nome_area, dados_area in dados_unidade["areas"].items():
            st.subheader(f"📌 {nome_area}")
            
            # Criar botões para cada link da área
            # Distribuir em 2 colunas para não ficar muito longo
            col1, col2 = st.columns(2)
            
            # Pegar todos os links da área
            links_items = list(dados_area["links"].items())
            
            # Distribuir os links entre as duas colunas
            for idx, (nome_link, url) in enumerate(links_items):
                coluna_alvo = col1 if idx % 2 == 0 else col2
                
                with coluna_alvo:
                    # Determinar ícone baseado no tipo de área
                    icone = {
                        "Gente": "👥",
                        "Segurança": "🛡️",
                        "Frota": "🚛",
                        "Distribuição": "📦",
                        "Armazem": "🏭",
                        "Financeiro": "💰",
                        "Gestão": "📊",
                        "Ajudante": "👷",
                        "Operador": "🔧",
                        "Manutenção": "🔨",
                        "Abastecimento": "⛽"
                    }.get(nome_link, "🔗")
                    
                    # Botão com link
                    if st.button(f"{icone} {nome_link}", key=f"{nome_unidade}_{nome_area}_{nome_link}", use_container_width=True):
                        # Abrir link em nova aba usando JavaScript
                        js = f"window.open('{url}')"
                        st.markdown(f'<script>{js}</script>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Espaço entre áreas
        
        st.markdown('</div>', unsafe_allow_html=True)

# Página principal
st.title("📚 Central de Materiais por Unidade")
st.markdown("---")

# Organizar unidades em linhas de 2 cards
unidades_lista = list(UNIDADES.items())

for i in range(0, len(unidades_lista), 2):
    # Criar linha com 2 colunas
    cols = st.columns(2)
    
    # Primeira unidade da linha
    with cols[0]:
        if i < len(unidades_lista):
            nome_unidade, dados_unidade = unidades_lista[i]
            criar_card_unidade(nome_unidade, dados_unidade)
    
    # Segunda unidade da linha
    with cols[1]:
        if i + 1 < len(unidades_lista):
            nome_unidade, dados_unidade = unidades_lista[i + 1]
            criar_card_unidade(nome_unidade, dados_unidade)
    
    st.markdown("---")  # Separador entre linhas

# Rodapé com informações
st.markdown("---")
st.caption("🔄 Clique nos botões para acessar os materiais. Os links abrirão em novas abas.")

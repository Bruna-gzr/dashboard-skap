# 1_Painel de Acessos RH.py
import streamlit as st

# Configuração da página
st.set_page_config(page_title="Materiais de Gestão RH", layout="wide")

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
    }

    .unidade-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .unidade-titulo {
        text-align: center;
        color: white;
        font-size: 18px;
        font-weight: 700;
        margin: 0;
        padding: 0;
        margin-bottom: 20px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #2D2D2D 0%, #404040 100%);
        border-radius: 20px;
        padding: 20px 15px 20px 15px;
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
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px auto;
    }
    
    .logo-fallback-grande {
        background: white;
        border-radius: 50%;
        width: 120px !important;
        height: 120px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px auto;
    }
    
    .logo-fallback span, .logo-fallback-grande span {
        font-size: 50px;
    }

    .link-botao {
        display: block;
        width: 100%;
        background: #3A3A3A;
        color: #FFFFFF !important;
        border: 1px solid #555555;
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 12px;
        font-weight: 500;
        text-align: left;
        text-decoration: none !important;
        margin: 5px 0;
        box-sizing: border-box;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
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
        padding: 8px 10px;
        font-size: 12px;
        font-weight: 500;
        text-align: left;
        text-decoration: none !important;
        margin: 5px 0;
        box-sizing: border-box;
        cursor: default;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    @media (max-width: 1200px) {
        .link-botao, .link-botao-vazio {
            font-size: 11px;
            padding: 6px 8px;
            white-space: normal;
            word-break: break-word;
        }
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown("<h1 class='page-title'>📊 MATERIAIS DE GESTÃO RH</h1>", unsafe_allow_html=True)

# ============================================
# DADOS DAS UNIDADES
# ============================================

UNIDADES = {
    "Londrina": {"logo": "logos/Londrina.png"},
    "Fco Beltrao": {"logo": "logos/Fco Beltrao.png"},
    "Diadema": {"logo": "logos/Diadema.png"},
    "Sao Cristovao": {"logo": "logos/Sao Cristovao.png"},
    "Cascavel": {"logo": "logos/Cascavel.png"},
    "Litoral": {"logo": "logos/Litoral.png"},
    "Foz do Iguacu": {"logo": "logos/Foz do Iguacu.png"},
    "Petropolis": {"logo": "logos/Petropolis.png"},
    "Ponta Grossa Empurrada": {"logo": "logos/Ponta Grossa Armazem.png"},
    "Ponta Grossa Armazem": {"logo": "logos/Ponta Grossa Armazem.png"},
    "Vidros": {"logo": "logos/Vidros.png"}
}

# Ícones para cada botão
ICONES = {
    "Comitê de Gente": "👥",
    "RPS de gente": "📋",
    "Café com Gerente": "☕",
    "Pipeline": "📊",
    "Matriz de Habilidades": "📚",
    "Check List DPO/VPO": "✅",
    "Defesa DPO/VPO": "🛡️"
}

# Botões padrão para todas as unidades (6 botões)
BOTOES_PADRAO = ["Comitê de Gente", "RPS de gente", "Café com Gerente", "Pipeline", "Check List DPO/VPO", "Defesa DPO/VPO"]

# Botões extras apenas para Ponta Grossa
BOTOES_PONTA_GROSSA = ["Comitê de Gente", "RPS de gente", "Café com Gerente", "Pipeline", "Matriz de Habilidades", "Check List DPO/VPO", "Defesa DPO/VPO"]

UNIDADES_LOGO_GRANDE = ["Litoral", "Vidros"]

# ============================================
# LINKS POR UNIDADE
# ============================================

LINKS = {
    "Londrina": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1NXMSxZqpvi5iUvmasP36feQqKJ_rgkTgAthy-qHuJaA/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1cnUILTRAHvMkJcqoFqgi4tA3kcx1c_0vjYg4t8mngV4/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/1lpNNIX54Cqz3H6wfOor5MjsA3iy1Rfqw7Ia7dSAVAM4/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1i0bnEazziVStCcSM71eoXMAPDrOiZhQEq9OVsrcsttM/edit?gid=499267061#gid=499267061",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1n7-Y0Fo5GpWSLeTGYuG5cfLkAN83hyygy0FAmlUb9fU/edit?gid=335667020#gid=335667020",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1LklxIqqIIs0RnvWWyLCFLM6N9WhdweJlXFFIgRIm2M0/edit?gid=1108610351#gid=1108610351"
    },
    "Fco Beltrao": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1P36JF03YRt__P-HBOYRiqAJQnLJKFyvLX5QhROKVrq4/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1d_qaye5LN7sQ_V3fYwHBGcDxTdyHk88_PL94NrJre5w/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/1rbQWzLP0DsHftjthjczuEjRevnIvukI7kXTlGBYKrMU/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/15R9xHMHruMXKon0lcU167pMHwSYO6brWJX6XocjRxzA/edit?gid=978593831#gid=978593831",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1adzXsFK3Ua3XBCOutg1qTA-wNES3bNpdJE-eANrv5GA/edit?gid=436320416#gid=436320416",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1w6Qakawbi44nLobZrZ-sk6Ms-1GBYJKInTccabtO8G0/edit?gid=1108610351#gid=1108610351"
    },
    "Diadema": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1Scebccspv23dWmEViZY-_r6oev9y6PEltF53S9QIbf0/edit?gid=690778771#gid=690778771",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1zdjilZtvJ3icWk37xMmyN6QcCMWtD_LDadlKIeREhGM/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/13Vq_tWhx8UDin3_r7bsb3BkY8flMyMgJ-j0qnRcc_Dw/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1j2hE5xM-MBocSpMnHZ2R_zdlqch5oGAE44IvY4KZyyk/edit?gid=978593831#gid=978593831",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1G4ocfhmWK3qeZ-HLjUEyd_0zvoLfCxn9vzcdxtqZ9Rc/edit?gid=875758865#gid=875758865",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1-Uv6WkuagTHL45U3Llnxm5CJ4IWtsEkmB0ZutN8pAnI/edit?gid=605770282#gid=605770282"
    },
    "Sao Cristovao": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1yON6JWVrJI3b36SdHA97DIP9yffSx3pRomguKcaIEXU/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1ulcKxVCxiabEt534Kmbf0r6lGop4gJmh2Vpnhto7KpQ/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/1JQ5WsjRd4DK5nj2fD2g3bNrpRF6mMus8iZX6wwze4-w/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1R02IYjnJJ86ZOOElNxXg6ac-6R55I22FMsZqofruo10/edit?gid=978593831#gid=978593831",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1TX9stGKqbvZGcbswfhGWw_Bg4JZuDmosCGkOJdoJytc/edit?gid=1810158702#gid=1810158702",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/15rbwRTqvqsAtKVqBu2dL3Xg7tjwzkwjkpAqHVBu5WKQ/edit?gid=1108610351#gid=1108610351"
    },
    "Cascavel": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1w-g58r0_kg4uIHpU3gd39RcAY-t_0RRPPmH1Jse5f8g/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1gKYYXkjmXxn0vkzp0F8AiutdDkUgY5E-9T37JeCi8V8/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/10o7xI9ZMXu8mkwC0zmD4x8Tnm4z97ku60PVTswpAsGw/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1n0TC6PybFOLpOqXr8r5CfOGap59eLTfbfhmaotpu0Vc/edit?gid=978593831#gid=978593831",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/16aa1EjeAlmYZieK6aq83wu04pjM_ppaK6retFz13cgM/edit?gid=107931427#gid=107931427",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1UG8xvVXwFKmzFrr_tXwJm-PBJ2U-n8jrpVCvKIZayEY/edit?gid=1108610351#gid=1108610351"
    },
    "Litoral": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1_BAluLZfN6QHvQrsWvJznEcsUvGxFBpFW258ay6mEuE/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1WwydxBRUypVi2oAdvGU0PWrTyzBAC4y8vIqtfG6VsAc/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/1zNagp5N38uThopnDCp_lBTEO7bXjj8CfR_kcx0geUMQ/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1o6Bkp8lCr41BXzo8wPUm6M3XNj2mp170vRhODN2EUO8/edit?gid=910918069#gid=910918069",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1Ucb1KmZSu3tH3IT5tNqkSIFnZqbsaLJypEppccMnTTA/edit?gid=875758865#gid=875758865",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1eNfFwwMKC9oYu0USeVb6-zSFHS1q79K2wU0aBTw9SVU/edit?gid=1108610351#gid=1108610351"
    },
    "Foz do Iguacu": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/18zB5tYT1hxtZFotpQU2ZnXz91UYypI3vmfhZKH8KHns/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1NN6HPaDF3FTvbEq4GKOpm8FfOGVCrDAWKVsOwyoj61g/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/1smVz8t0DjCVoi76OwJnCr7GnFsdjuJZ6KD62aVNxUEo/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1X3MWXRLCAUt9unenwxmIGZhdBwZoP1v4EDLBKMQIAoI/edit?gid=978593831#gid=978593831",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1JIYQpmkWJt77xGmIm3R5oqOI2YPIHsR0BbmWr4QH6y4/edit?gid=1523475353#gid=1523475353",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1MbIMt9FiZJBjU6VlgijsQeqLB7YFfc-nlsFK1P78uCs/edit?gid=1108610351#gid=1108610351"
    },
    "Petropolis": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1UJU5GjR8520gTfSoz0bCjDGOMvZavKxRV4Fp_U--7Ho/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1UG5OINGR6ftx52dJAy_sLHfI_AhDxOMaiClLCi6wWbc/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/1bEPBZiYinm8DRFwg3YiHbxxqSpavBZCyu6K_cXTKQag/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1nbmkzFAV9fuq6N0LmlThncsLhUxJU6QacAKu0OPj840/edit?gid=910918069#gid=910918069",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1Q29I79L2yiF_u9D9xODPV818P-pxV-PLblT5FzzjVBs/edit",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1XbcX3L-eQcr8mhyqW9kHgwye3yzU4iu2Vab7to5afC0/edit?gid=1108610351#gid=1108610351"
    },
    "Ponta Grossa Empurrada": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1mBExhtRaHTZXRb_aIXpg4OZdmMne_BADmFeUag2LdXM/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1X749vXsiLwj29r8v21GsoYe33ZnpQRPCC8esDWEPXRk/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/14otbkOzL9n1yyCmLXNl6VlpTwuFlbyUpjzWMuL-FX8o/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1nlWF_HwtK-7uz_bd3O_pYmImm5ZTn7ay3k7Rz0gqIkU/edit?gid=1754165725#gid=1754165725",
        "Matriz de Habilidades": "https://docs.google.com/spreadsheets/d/1LMQQTJyZVazt-8qQwecd_QAjZ-l4lqDXIl9k6uBbHAE/edit?gid=613075840#gid=613075840",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1SmalAzLMKEO_6CjapqYjGQbo-So2LRfl3DT77HnKZpc/edit?gid=1146159521#gid=1146159521",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/169N0avHeBWROuaozVHzACBJZ1REjDlI1jLFIlrxC10E/edit?gid=1108610351#gid=1108610351"
    },
    "Ponta Grossa Armazem": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1IHFuQGoPp4OfPGBl5KWtPFHDr6Gl_fRjORx00pcmaY4/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1h219cyeRrpx2LdMA6GHJrEG31Wu6gEMdZukbMvAkU_k/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/1hPzkB3ByK-O7mP5NAduzhpAWeUgSGxZBV_SQ6Yw8U8A/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/10Gx5jXx2z_9F8YAXRna9YsHOfvvkwJ0Kw9p6MMGXtmQ/edit?gid=499267061#gid=499267061",
        "Matriz de Habilidades": "https://docs.google.com/spreadsheets/d/1H5uVPMGmPwBy7_YUGWvMoLnjQ6b_6GjooLN_yRWkxmI/edit?gid=0#gid=0",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1EcW0G7i3wbmQ8ocdm2vHjG8P5J0wBUgZGawE_Otflmk/edit?gid=1146159521#gid=1146159521",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1FTXZe9EqDDCTK_jCHiArWL9kKyPw8ZHzRPJBEtZH7D4/edit?gid=1108610351#gid=1108610351"
    },
    "Vidros": {
        "Comitê de Gente": "https://docs.google.com/spreadsheets/d/1fmfUf3SNK3sWIojBrDsQv9FsHovVra1RXi4XkpuJzeI/edit?gid=915035701#gid=915035701",
        "RPS de gente": "https://docs.google.com/spreadsheets/d/1TYzUjg6HJttW9pUCjn0WkfvuT1UYre7QwHJJE_R8EYY/edit?gid=1903627627#gid=1903627627",
        "Café com Gerente": "https://docs.google.com/spreadsheets/d/1Jr43BKk5jR9oX8l8Vv1Tx3aa8PHLvepq_NCfqo9U29o/edit?gid=365657887#gid=365657887",
        "Pipeline": "https://docs.google.com/spreadsheets/d/1-9vF4G5NMjM-pfdQg9wXempoEwRsxqAvz8irJwvyZjw/edit?gid=499267061#gid=499267061",
        "Check List DPO/VPO": "https://docs.google.com/spreadsheets/d/1PJYyMTY_aO1vSbOlrXNKDnRix7b_n0haQ5zCuGI-38k/edit?gid=335667020#gid=335667020",
        "Defesa DPO/VPO": "https://docs.google.com/spreadsheets/d/1r8QRVSHYM86RoOQoKblhLLeYmaUiGQrw1LwJAOp73aU/edit?gid=1108610351#gid=1108610351"
    }
}

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
        logo_width = 120 if logo_grande else 100
        
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

        # Define quais botões usar baseado na unidade
        if nome_unidade in ["Ponta Grossa Empurrada", "Ponta Grossa Armazem"]:
            botoes = BOTOES_PONTA_GROSSA
        else:
            botoes = BOTOES_PADRAO
        
        for botao in botoes:
            icone = ICONES.get(botao, "🔗")
            link = LINKS.get(nome_unidade, {}).get(botao, "")
            render_link_botao(f"{icone} {botao}", link)

# Layout com 4 colunas
unidades_lista = list(UNIDADES.items())
NUM_COLUNAS = 4

for i in range(0, len(unidades_lista), NUM_COLUNAS):
    cols = st.columns(NUM_COLUNAS, gap="medium")
    
    for j in range(NUM_COLUNAS):
        idx = i + j
        if idx < len(unidades_lista):
            with cols[j]:
                nome, dados = unidades_lista[idx]
                criar_card_unidade(nome, dados)

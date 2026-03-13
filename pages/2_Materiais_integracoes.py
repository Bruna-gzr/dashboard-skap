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

    .titulo-coluna {
        color: #CCCCCC;
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 16px;
        text-align: center;
        width: 100%;
        letter-spacing: 0.5px;
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
        padding: 8px 12px;
        font-size: 14px;
        font-weight: 500;
        text-align: left;
        text-decoration: none !important;
        margin: 3px 0;
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
        padding: 8px 12px;
        font-size: 14px;
        font-weight: 500;
        text-align: left;
        text-decoration: none !important;
        margin: 3px 0;
        box-sizing: border-box;
        cursor: default;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown("<h1 class='page-title'>MATERIAIS DE INTEGRAÇÃO</h1>", unsafe_allow_html=True)

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
            "setores": ["GENTE", "SEGURANÇA", "OPERAÇÃO"]
        },
        "coluna2": {
            "titulo": "👷🏻‍♂️ ARMAZEM",
            "setores": ["GESTÃO", "GENTE", "SEGURANÇA", "AJUDANTE DE ARMAZEM", "OPERADOR", "FROTA"]
        }
    },
    "Sao Cristovao": {
        "logo": "logos/Sao Cristovao.png",
        "coluna1": {
            "titulo": "🚛 DISTRIBUIÇÃO",
            "setores": ["GENTE", "SEGURANÇA", "ENTREGA", "FINANCEIRO", "FROTA", "GESTÃO"]
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

ICONES = {
    "GENTE": "👥",
    "SEGURANÇA": "🛡️",
    "ENTREGA": "🚚",
    "OPERAÇÃO": "⚙️",
    "FINANCEIRO": "💰",
    "FROTA": "🚛",
    "GESTÃO": "📊",
    "AJUDANTE DE ARMAZEM": "📦",
    "OPERADOR": "🔧"
}

UNIDADES_LOGO_GRANDE = ["Litoral", "Vidros"]

LINKS = {
    "Cascavel": {
        "DISTRIBUIÇÃO": {
            "GENTE": "https://www.canva.com/design/DAGU-QmNebQ/8Kw0euGLNWLECVYFqsQdMQ/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVDAaKz2M/OD4i20ecobAgfKMzXc6UWw/edit",
            "ENTREGA": "https://www.canva.com/design/DAGVDYC9E8A/XjP1QfpAly6vh5GMa4qHeg/edit",
            "FINANCEIRO": "https://www.canva.com/design/DAGVEdX3TZ8/NE2n7F8TJWoKSBAzJcN4uw/edit",
            "FROTA": "https://www.canva.com/design/DAGSmyTxPe0/aQBjPT4Cl1eSKOy8syqS3w/edit",
        },
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGU_LWGMcI/WfOjVdHq0Z0VXL2k5NU66g/edit",
            "GENTE": "https://www.canva.com/design/DAGU_B8yDeY/hFDGei0Kk4kB_yUU3FV-Ww/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVVu5cZKg/pFcm4LK9x-bIzyVUbQ2dTA/edit",
            "AJUDANTE DE ARMAZEM": "https://www.canva.com/design/DAGVVxse6P8/ZK5IbaiyDiCiujblgzqrlA/edit",
            "OPERADOR": "https://www.canva.com/design/DAGVVrreB8k/pqquFJQh8iol8TFxUS4WeQ/edit",
        }
    },
    "Fco Beltrao": {
        "DISTRIBUIÇÃO": {
            "GENTE": "https://www.canva.com/design/DAGU-XjNPLY/hA4xTLm0tTVWoId2Zo-C2g/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVDKtWIs0/uxqPHW79vJFlhwalB1Oq4Q/edit",
            "ENTREGA": "https://www.canva.com/design/DAGVDc36cfM/IsTMyzrBlcij9_QwhDUkEg/edit",
            "FINANCEIRO": "https://www.canva.com/design/DAGVEd4XEG8/_ctUM32RwNdcxIUkSPs2HA/edit",
            "FROTA": "https://www.canva.com/design/DAGS41NTZxw/4HrzgYFfkORPH2bzZQW5SA/edit",
        },
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGU_HP0aYk/sA3bds7hG4thnclb9ZSgFg/edit",
            "GENTE": "https://www.canva.com/design/DAGU_FyMpic/VuGGBqWhcfO3FjN0HjY77A/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVVo4nEFI/7sD5YovYS4Ju-bb_RDwN3g/edit",
            "AJUDANTE DE ARMAZEM": "https://www.canva.com/design/DAGVVwrAEIY/ehdx-uslpTN1PA31kBrH1w/edit",
            "OPERADOR": "https://www.canva.com/design/DAGVVsN6DnE/ISHJuq1sp9V13gxaOxPekQ/edit",
        }
    },
    "Foz do Iguacu": {
        "DISTRIBUIÇÃO": {
            "GENTE": "https://www.canva.com/design/DAGU-YEvvSE/jJMcuVGw0ZdIqfQ8w0robg/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVDJ_n-f8/49pIU2gWBObd1wsiajBxEw/edit",
            "ENTREGA": "https://www.canva.com/design/DAGVDTtzpIE/IIA1qsi03Q1RELHdVBQdvA/edit",
            "FINANCEIRO": "https://www.canva.com/design/DAGVEbX_oV4/kG1bjIY0o6AlhWqpqcfyEw/edit",
            "FROTA": "https://www.canva.com/design/DAGS5F4ogA4/iUo8W7htTmWFfy59Tdo2FA/edit",
        },
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGU_HWh1sA/U7wbr06VOK0iSGPuwL600Q/edit",
            "GENTE": "https://www.canva.com/design/DAGU_JRutl8/J4p8GTKblzYTa-7Je9e_UQ/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVVp_9zt8/aenfN2xqFCeDhiO_kazEiQ/edit",
            "AJUDANTE DE ARMAZEM": "https://www.canva.com/design/DAGVV8zXKHY/dNDjOJuwBcQjiqYLA2CqJA/edit",
            "OPERADOR": "https://www.canva.com/design/DAGVVqiIATU/eXCgLPXefRAag1HkG2c0gg/edit",
        }
    },
    "Londrina": {
        "DISTRIBUIÇÃO": {
            "GENTE": "https://www.canva.com/design/DAGU-QimPoo/AGLhPJDzCJeydJVc3yd0Uw/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGxwfnsKiU/Icny_PG98jJHP7FWLmVhwQ/edit",
            "ENTREGA": "https://www.canva.com/design/DAGVDRfGt_0/07WhSiMvJVpvvbi5hR1zoA/edit",
            "FINANCEIRO": "https://www.canva.com/design/DAGVEYvZH9I/dR645fh3d4yuRjLy0rYO7A/edit",
            "FROTA": "https://www.canva.com/design/DAGS5OWwKZs/vMpW_RwRkZU8C8AJXyFl6g/edit",
        },
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGU_C4UVDg/arhI8vC22LScNBOma3v9Dg/edit",
            "GENTE": "https://www.canva.com/design/DAGU_MugIS4/zbsmu8cr1xiEdBysOLLSCw/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVVu_kGa0/__pCEmCPdzQTzupz1tQpgg/edit",
            "AJUDANTE DE ARMAZEM": "https://www.canva.com/design/DAGVV_9hm_g/QozEV7PX_sed9KRCvvvy7g/edit",
            "OPERADOR": "https://www.canva.com/design/DAGVVgd3aGY/P8dn9WB_DhgUUPWZ_Vy_jA/edit",
        }
    },
    "Diadema": {
        "DISTRIBUIÇÃO": {
            "GENTE": "https://www.canva.com/design/DAGU-viAM-w/knESAP3LfoxcLYdzwf3RHQ/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVDJTHnUI/w30aGKH-wifdczxsibtulg/edit",
            "ENTREGA": "https://www.canva.com/design/DAGVDVsIO_U/trnm_YPEDyIT2RJk5jAfSA/edit",
            "FINANCEIRO": "https://www.canva.com/design/DAGVESit5Xo/pUWNe0NGO9D_ZADagWF17A/edit",
            "FROTA": "https://www.canva.com/design/DAGHYolhfOU/K-VpsGPOIyE2o3tEQgLHxQ/edit",
        },
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGU_OL2cqc/GcC9rskk1Ftyxgy-udH6Rg/edit",
            "GENTE": "https://www.canva.com/design/DAGU_EgSLXU/2mgHz7FfnJXMC3kHDk9wEw/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVVp-v6RY/qL9cHSTnPQh17M3sX_Cj-A/edit",
            "AJUDANTE DE ARMAZEM": "https://www.canva.com/design/DAGUUNaGJb0/wdNaRTsmoOyo0yzqR8BbEQ/edit",
            "OPERADOR": "https://www.canva.com/design/DAGVVq9lo1I/ou8Zc2jRQyoFj2rJa12B0w/edit",
        }
    },
    "Litoral": {
        "DISTRIBUIÇÃO": {
            "GENTE": "https://www.canva.com/design/DAGU-_M4PNs/DtCfh1gg32RxM7OrvCxPsA/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVDK8SlYo/mHhTWjaswJ6bwPNdtZ_PNA/edit",
            "ENTREGA": "https://www.canva.com/design/DAGVDUU7vxc/tf6_fp5rfd5RIZhHlM_3DQ/edit",
            "FINANCEIRO": "https://www.canva.com/design/DAGVEf6lCn8/Xzj9ZW82UqXegUYR3Ddg7g/edit",
            "FROTA": "https://www.canva.com/design/DAGS48PUqZg/cHO-63N_1MQsUi_9TQP1Gw/edit",
        },
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGVDISJ7t8/eBxco-geG9dzcUscuN4Ajw/edit",
            "GENTE": "https://www.canva.com/design/DAGU_CP0NZM/ah6bOwQYckzPHhKyokgZCA/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVVtuVYCg/3LiiUUZ3g7oFPob03F7Q_A/edit",
            "AJUDANTE DE ARMAZEM": "https://www.canva.com/design/DAGVV0Aki6Q/a3CzTux8yUK9Eu2rRDtzjQ/edit",
            "OPERADOR": "https://www.canva.com/design/DAGVVrEn0GU/n4litgTm7Wkd73GP_CDv9w/edit",
        }
    },
    "Sao Cristovao": {
        "DISTRIBUIÇÃO": {
            "GENTE": "https://www.canva.com/design/DAGU-px67Ug/sg0N-xgn0mrTwEpS7vBGCA/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGVDCUZENQ/VsxePkCU4LSgVBXtucWUmA/edit",
            "ENTREGA": "https://www.canva.com/design/DAGVDRf3UEs/jfFvTjmW82rZbagUM84lWQ/edit",
            "FINANCEIRO": "https://www.canva.com/design/DAGVEWaYVHk/HUHW9SpcJ4dxu2xlGULANg/edit",
            "FROTA": "https://www.canva.com/design/DAGS5PauqJw/06ATl-2-QoI9gkzLe6jf6A/edit",
            "GESTÃO": "https://www.canva.com/design/DAGU_JUK-OU/PjBIdp1pSyk4WCS7hpW1eA/edit",
        }
    },
    "Petropolis": {
        "DISTRIBUIÇÃO": {
            "GENTE": "https://www.canva.com/design/DAGrdKuUfKo/-i30IT8uP8vVVSyDw3QVew/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGrdLiDQEw/DZMkbDlZ0H3iFwBrhrgB_g/edit",
            "ENTREGA": "https://www.canva.com/design/DAGrdMgj6DA/KnEjbwN-4rl1CduhUO6lJg/edit",
            "FINANCEIRO": "https://www.canva.com/design/DAGrdEdcYN4/dymXcVB8yUZkVUJyyu5jZw/edit",
            "FROTA": "https://www.canva.com/design/DAGrdIoLAjA/sR7gtw6yefGLh9-Yw30IpA/edit",
        },
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGrdBYuwXw/aDV82dlEboasiO4S80r7WA/edit",
            "GENTE": "https://www.canva.com/design/DAGrdJnnOyc/hYFCZOdwjt1zdNgS3tcFeg/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGrdG5JKAs/ooUa--ecof6rbjHUZzlbWg/edit",
            "AJUDANTE DE ARMAZEM": "https://www.canva.com/design/DAGrdAuS2kM/10_RN98QD1R7YtVw9RdgYw/edit",
            "OPERADOR": "https://www.canva.com/design/DAGrdPW5PYI/jV0Np_pbDfIzz54Th11e8Q/edit",
        }
    },
    "Ponta Grossa": {
        "EMPURRADA": {
            "GENTE": "https://www.canva.com/design/DAG13LIpHqU/uyorO3CZAB4L5a5j-285KA/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGvfQn9TOM/KfTOeAQ0jykfbcOMuFitxQ/edit",
            "OPERAÇÃO": "https://www.canva.com/design/DAG1yu-k6hE/OMLmz0AZg2TWe3ePQDQ4kw/edit",
        },
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGzPso2Yvo/N-AeSPyW6kmpdGb1iGk_pw/edit",
            "GENTE": "https://www.canva.com/design/DAGzPoLU4gk/Gu0nh6pvWZrHKxJWx5w9eg/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGzRb9QDkQ/scKlpyHBByhPlWjwDPIWPQ/edit",
            "AJUDANTE DE ARMAZEM": "https://www.canva.com/design/DAGzPsb89EU/5bOFnfZFvH8dpaqFnEewiA/edit",
            "OPERADOR": "https://www.canva.com/design/DAGzPtRpVhU/M2A2ILVZhDCrDkk_8sbgew/edit",
            "FROTA": "https://www.canva.com/design/DAGzPhjh23c/Freun41VTIEhg0O3g_Ab2A/edit",
        }
    },
    "Vidros": {
        "ARMAZEM": {
            "GESTÃO": "https://www.canva.com/design/DAGzPso2Yvo/N-AeSPyW6kmpdGb1iGk_pw/edit",
            "GENTE": "https://www.canva.com/design/DAGzwHPQ1dw/cgSARPRHdtYGjc2MkGm0Vg/edit",
            "SEGURANÇA": "https://www.canva.com/design/DAGWAUbDBuw/ZuogV5Rf5zh2eqxIhd11nw/edit",
            "AJUDANTE DE ARMAZEM": "",
            "OPERADOR": "",
        }
    }
}

def buscar_link(nome_unidade, titulo_coluna, setor):
    chave_coluna = (
        titulo_coluna.replace("🚛 ", "")
        .replace("👷🏻‍♂️ ", "")
        .replace("👷 ", "")
        .strip()
    )
    return LINKS.get(nome_unidade, {}).get(chave_coluna, {}).get(setor, "")

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

        colunas_ativas = 0
        if dados["coluna1"] is not None:
            colunas_ativas += 1
        if dados["coluna2"] is not None:
            colunas_ativas += 1
        
        if colunas_ativas == 2:
            col1, col2 = st.columns(2)
            
            with col1:
                titulo_coluna_1 = dados["coluna1"]["titulo"]
                st.markdown(
                    f"<div class='titulo-coluna'>{titulo_coluna_1}</div>",
                    unsafe_allow_html=True
                )
                for setor in dados["coluna1"]["setores"]:
                    icone = ICONES.get(setor, "🔗")
                    link = buscar_link(nome_unidade, titulo_coluna_1, setor)
                    render_link_botao(f"{icone} {setor}", link)

            with col2:
                titulo_coluna_2 = dados["coluna2"]["titulo"]
                st.markdown(
                    f"<div class='titulo-coluna'>{titulo_coluna_2}</div>",
                    unsafe_allow_html=True
                )
                for setor in dados["coluna2"]["setores"]:
                    icone = ICONES.get(setor, "🔗")
                    link = buscar_link(nome_unidade, titulo_coluna_2, setor)
                    render_link_botao(f"{icone} {setor}", link)
        
        elif colunas_ativas == 1:
            col = st.columns([1, 2, 1])[1]
            
            with col:
                if dados["coluna1"] is not None:
                    titulo_coluna = dados["coluna1"]["titulo"]
                    st.markdown(
                        f"<div class='titulo-coluna'>{titulo_coluna}</div>",
                        unsafe_allow_html=True
                    )
                    for setor in dados["coluna1"]["setores"]:
                        icone = ICONES.get(setor, "🔗")
                        link = buscar_link(nome_unidade, titulo_coluna, setor)
                        render_link_botao(f"{icone} {setor}", link)
                
                elif dados["coluna2"] is not None:
                    titulo_coluna = dados["coluna2"]["titulo"]
                    st.markdown(
                        f"<div class='titulo-coluna'>{titulo_coluna}</div>",
                        unsafe_allow_html=True
                    )
                    for setor in dados["coluna2"]["setores"]:
                        icone = ICONES.get(setor, "🔗")
                        link = buscar_link(nome_unidade, titulo_coluna, setor)
                        render_link_botao(f"{icone} {setor}", link)

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

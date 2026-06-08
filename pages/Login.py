# Login.py - Tela de login e menu principal
import streamlit as st
import sys
from pathlib import Path

# Adicionar diretório atual ao path
sys.path.insert(0, str(Path(__file__).parent))

# Importar sistema de login
from login_central import verificar_login, exibir_info_usuario_sidebar, get_unidade_usuario, get_usuario

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Login - Dashboards",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# VERIFICAR LOGIN (mostra a tela se não estiver logado)
# =========================
if not verificar_login():
    st.stop()  # Fica na tela de login até autenticar

# Se chegou aqui, está logado!
exibir_info_usuario_sidebar()

# =========================
# TÍTULO PRINCIPAL (pós-login)
# =========================
st.markdown("# 🏢 Sistema de Dashboards Corporativos")
st.markdown(f"### Bem-vindo, **{get_usuario()}**!")
st.markdown(f"**📍 Unidade:** {get_unidade_usuario()}")
st.markdown("---")

# =========================
# MENU DE DASHBOARDS
# =========================
st.subheader("📊 Selecione o Dashboard:")

# Lista dos seus dashboards
dashboards = {
    "📦 Integração Armazém": "dashboards/integracao_armazem.py",
    "👨🏻‍🎓 Gestão de Padrinhos": "dashboards/gestao_padrinhos.py",
    "📈 Raio X": "dashboards/raio_x.py",
    "📊 Skap": "dashboards/skap.py",
    "🚚 Integração Distribuição": "dashboards/integracao_distribuicao.py",
}

# Botões em grid 2x3
cols = st.columns(2)
for idx, (nome, caminho) in enumerate(dashboards.items()):
    with cols[idx % 2]:
        if st.button(nome, use_container_width=True, key=f"btn_{idx}"):
            st.session_state.dashboard_selecionado = caminho
            st.rerun()

# =========================
# EXECUTAR DASHBOARD SELECIONADO
# =========================
if "dashboard_selecionado" in st.session_state:
    st.markdown("---")
    st.subheader("📋 Dashboard:")
    
    try:
        # Executar o dashboard escolhido
        with open(st.session_state.dashboard_selecionado, "r", encoding="utf-8") as file:
            exec(file.read())
    except Exception as e:
        st.error(f"Erro ao carregar dashboard: {e}")
        st.code(f"Detalhes: {str(e)}")

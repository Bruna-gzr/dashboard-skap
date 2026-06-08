# app.py
import streamlit as st
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime

# =========================
# FUNÇÕES DE LOGIN
# =========================

def hash_senha(senha: str) -> str:
    return hashlib.sha256(senha.encode()).hexdigest()

@st.cache_data(ttl=3600)
def carregar_credenciais():
    caminhos = [
        Path(".data/credenciais.xlsx"),
        Path("data/credenciais.xlsx"),
        Path("credenciais.xlsx"),
    ]
    for caminho in caminhos:
        if caminho.exists():
            return pd.read_excel(caminho)
    
    # Credenciais padrão para teste
    return pd.DataFrame([
        {"Usuario": "Adm", "Senha": hash_senha("Adm@log20@"), "Operacao": "Todas"},
        {"Usuario": "cd.litoral", "Senha": hash_senha("lito@log20"), "Operacao": "CD LITORAL"},
        {"Usuario": "cd.petrop", "Senha": hash_senha("petr@log20"), "Operacao": "CD PETRÓPOLIS"},
    ])

def fazer_login():
    if "logado" in st.session_state and st.session_state.logado:
        return True
    
    st.markdown("""
    <style>
    .login-box {
        max-width: 400px;
        margin: 100px auto;
        padding: 30px;
        background: #1e1e2e;
        border-radius: 15px;
        border: 1px solid #f0d36b;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown("## 🔐 Login")
    
    with st.form("login"):
        usuario = st.text_input("Usuário")
        senha = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar", use_container_width=True)
        
        if submitted:
            creds = carregar_credenciais()
            senha_hash = hash_senha(senha)
            user = creds[(creds["Usuario"] == usuario) & (creds["Senha"] == senha_hash)]
            
            if not user.empty:
                st.session_state.logado = True
                st.session_state.usuario = user.iloc[0]["Usuario"]
                st.session_state.operacao = user.iloc[0]["Operacao"]
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return False

def get_operacao():
    return st.session_state.get("operacao", "Todas")

def get_usuario():
    return st.session_state.get("usuario", "")

def aplicar_filtro(df, coluna="Operação"):
    if df is None or df.empty:
        return df
    operacao = get_operacao()
    if operacao == "Todas" or get_usuario() == "Adm":
        return df
    if coluna in df.columns:
        return df[df[coluna].astype(str).str.strip() == operacao].copy()
    return df

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Painel Gente & Gestão",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# VERIFICAR LOGIN
# =========================
if not fazer_login():
    st.stop()

# =========================
# MENU PRINCIPAL (pós-login)
# =========================

# Sidebar com info do usuário
st.sidebar.markdown("---")
st.sidebar.markdown(f"**👤 Usuário:** {get_usuario()}")
st.sidebar.markdown(f"**📍 Operação:** {get_operacao()}")

if st.sidebar.button("🚪 Sair"):
    for key in ["logado", "usuario", "operacao"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Título principal
st.title("🏠 Home – Painel Gente & Gestão")
st.write(f"Bem-vindo, **{get_usuario()}**!")
st.write("Use os botões abaixo para navegar entre os módulos.")

st.markdown("---")
st.subheader("📊 Dashboards Disponíveis")

# Botões para cada dashboard (usando .pages/)
col1, col2 = st.columns(2)

with col1:
    if st.button("🔐 Painel de Acessos RH", use_container_width=True):
        st.switch_page(".pages/1_Painel de Acessos RH.py")
    
    if st.button("📚 Materiais Integrações", use_container_width=True):
        st.switch_page(".pages/2_Materiais_integracoes.py")
    
    if st.button("🚚 Integração Distribuição", use_container_width=True):
        st.switch_page(".pages/3_Integracao_Distribuicao.py")
    
    if st.button("📦 Integração Armazém", use_container_width=True):
        st.switch_page(".pages/4_Integracao_Armazem.py")

with col2:
    if st.button("👨🏻‍🎓 Padrinhos", use_container_width=True):
        st.switch_page(".pages/4_Padrinhos.py")
    
    if st.button("📋 Acompanhamento de Novos", use_container_width=True):
        st.switch_page(".pages/Acompanhamento_de_Novos.py")
    
    if st.button("📈 Raio X", use_container_width=True):
        st.switch_page(".pages/Raio X Operação.py")
    
    if st.button("📊 SKAP", use_container_width=True):
        st.switch_page(".pages/SKAP.py")

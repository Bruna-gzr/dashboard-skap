import streamlit as st
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime

def hash_senha(senha: str) -> str:
    return hashlib.sha256(senha.encode()).hexdigest()

@st.cache_data(ttl=3600)
def carregar_credenciais():
    """Carrega a planilha de credenciais"""
    possiveis_caminhos = [
        Path("data/credenciais.xlsx"),
        Path("../data/credenciais.xlsx"),
        Path("./data/credenciais.xlsx"),
        Path(__file__).parent / "data" / "credenciais.xlsx",
    ]
    
    for caminho in possiveis_caminhos:
        if caminho.exists():
            return pd.read_excel(caminho)
    
    st.error("❌ Arquivo de credenciais não encontrado!")
    return pd.DataFrame()

def verificar_login():
    """Verifica login e mostra tela se necessário"""
    
    # Já está logado?
    if "logado" in st.session_state and st.session_state.logado:
        return True
    
    # CSS para a tela de login
    st.markdown("""
    <style>
    .login-container {
        max-width: 420px;
        margin: 80px auto;
        padding: 35px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        border: 1px solid #f0d36b;
    }
    .login-title {
        text-align: center;
        color: #f0d36b;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔐 Login</div>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            usuario = st.text_input("👤 Usuário")
            senha = st.text_input("🔒 Senha", type="password")
            submitted = st.form_submit_button("📥 Entrar", use_container_width=True)
            
            if submitted:
                if not usuario or not senha:
                    st.error("❌ Preencha usuário e senha")
                    return False
                
                credenciais = carregar_credenciais()
                if credenciais.empty:
                    return False
                
                senha_hash = hash_senha(senha)
                
                user_data = credenciais[
                    (credenciais["Usuario"].astype(str).str.lower() == usuario.lower()) & 
                    (credenciais["Senha"] == senha_hash)
                ]
                
                if not user_data.empty:
                    st.session_state.logado = True
                    st.session_state.usuario = user_data.iloc[0]["Usuario"]
                    st.session_state.operacao = user_data.iloc[0]["Operacao"]  # mudou para Operacao
                    st.session_state.login_time = datetime.now()
                    st.success(f"✅ Bem-vindo, {usuario}!")
                    st.rerun()
                    return True
                else:
                    st.error("❌ Usuário ou senha inválidos")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False

def exibir_info_usuario_sidebar():
    """Mostra info do usuário no sidebar"""
    if not st.session_state.get("logado", False):
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="background: #1e1e2e; padding: 12px; border-radius: 10px;">
        <div style="color: #f0d36b; font-weight: bold;">👤 Logado</div>
        <div style="font-size: 13px;">📛 {st.session_state.usuario}</div>
        <div style="font-size: 13px;">🏭 Operação: {st.session_state.operacao}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Sair", use_container_width=True):
        for key in ["logado", "usuario", "operacao", "login_time"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def get_operacao_usuario():
    """Retorna a operação do usuário logado"""
    return st.session_state.get("operacao", "Todas")

def get_usuario():
    return st.session_state.get("usuario", "")

def aplicar_filtro_operacao(df, coluna="Operação"):
    """Filtra DataFrame pela operação do usuário"""
    if df is None or df.empty:
        return df
    
    operacao = get_operacao_usuario()
    
    # Admin ou usuário sem operação específica vê tudo
    if not operacao or operacao == "Todas" or get_usuario() == "Adm":
        return df
    
    # Verifica se a coluna existe
    if coluna in df.columns:
        df_filtrado = df[df[coluna].astype(str).str.strip() == operacao].copy()
        return df_filtrado
    
    # Se não encontrou a coluna, retorna o original (com aviso)
    st.sidebar.warning(f"⚠️ Coluna '{coluna}' não encontrada neste DataFrame")
    return df

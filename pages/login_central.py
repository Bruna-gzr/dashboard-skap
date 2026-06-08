# login_central.py
import streamlit as st
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime

def hash_senha(senha: str) -> str:
    """Cria hash SHA256 da senha"""
    return hashlib.sha256(senha.encode()).hexdigest()

@st.cache_data(ttl=3600)
def carregar_credenciais():
    """Carrega a planilha de credenciais"""
    # Procura o arquivo em diferentes lugares
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
    st.info("Crie o arquivo 'data/credenciais.xlsx' com as colunas: Usuario, Senha, Unidade")
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
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    .login-title {
        text-align: center;
        color: #f0d36b;
        margin-bottom: 10px;
        font-size: 28px;
        font-weight: bold;
    }
    .login-subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 30px;
        font-size: 14px;
    }
    .login-footer {
        text-align: center;
        margin-top: 20px;
        color: #666;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Centralizar na tela
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔐 Login</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Acesso aos Dashboards Corporativos</div>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            usuario = st.text_input("👤 Usuário", placeholder="Digite seu usuário")
            senha = st.text_input("🔒 Senha", type="password", placeholder="Digite sua senha")
            submitted = st.form_submit_button("📥 Entrar", use_container_width=True)
            
            if submitted:
                if not usuario or not senha:
                    st.error("❌ Preencha usuário e senha")
                    return False
                
                credenciais = carregar_credenciais()
                if credenciais.empty:
                    return False
                
                senha_hash = hash_senha(senha)
                
                # Buscar usuário
                user_data = credenciais[
                    (credenciais["Usuario"].astype(str).str.strip().str.lower() == usuario.lower().strip()) & 
                    (credenciais["Senha"] == senha_hash)
                ]
                
                if not user_data.empty:
                    st.session_state.logado = True
                    st.session_state.usuario = user_data.iloc[0]["Usuario"]
                    st.session_state.unidade = user_data.iloc[0]["Unidade"]
                    st.session_state.login_time = datetime.now()
                    st.success(f"✅ Bem-vindo, {usuario}!")
                    st.rerun()
                    return True
                else:
                    st.error("❌ Usuário ou senha inválidos")
                    return False
        
        st.markdown('<div class="login-footer">🔒 Ambiente seguro - Acesso restrito</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False

def exibir_info_usuario_sidebar():
    """Mostra info do usuário no sidebar (já logado)"""
    if not st.session_state.get("logado", False):
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="background: #1e1e2e; padding: 12px; border-radius: 10px;">
        <div style="color: #f0d36b; font-weight: bold;">👤 Logado</div>
        <div style="font-size: 13px;">📛 {st.session_state.usuario}</div>
        <div style="font-size: 13px;">🏭 {st.session_state.unidade}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Sair", use_container_width=True):
        for key in ["logado", "usuario", "unidade", "login_time"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def get_unidade_usuario():
    return st.session_state.get("unidade", "Todas")

def get_usuario():
    return st.session_state.get("usuario", "")

def aplicar_filtro_unidade(df, coluna_sugerida="Operação"):
    """
    Filtra DataFrame pela unidade do usuário.
    Funciona com colunas chamadas: Operação, Unidade, CD, Filial, etc.
    """
    if df is None or df.empty:
        return df
    
    unidade = get_unidade_usuario()
    
    # Admin ou usuário sem unidade específica vê tudo
    if not unidade or unidade == "Todas" or get_usuario() == "Adm":
        return df
    
    # Lista de possíveis nomes de coluna (ordem de prioridade)
    possiveis_nomes = [
        "Operação", "Unidade", "CD", "Filial", 
        "OPERACAO", "UNIDADE", "operacao", "unidade",
        "Op", "OP", "op", "unid"
    ]
    
    # Primeiro, tenta encontrar a coluna exata
    coluna_encontrada = None
    
    # Tenta a coluna sugerida primeiro
    if coluna_sugerida in df.columns:
        coluna_encontrada = coluna_sugerida
    else:
        # Procura por qualquer nome na lista
        for nome in possiveis_nomes:
            if nome in df.columns:
                coluna_encontrada = nome
                break
        
        # Se ainda não encontrou, procura por similaridade (case-insensitive)
        if not coluna_encontrada:
            for col in df.columns:
                col_lower = col.lower()
                if any(palavra.lower() in col_lower for palavra in ["operacao", "unidade", "cd", "filial"]):
                    coluna_encontrada = col
                    break
    
    # Aplica o filtro se encontrou a coluna
    if coluna_encontrada:
        # Converte para string e faz a comparação
        mask = df[coluna_encontrada].astype(str).str.strip() == unidade
        
        # Se não achou exato, tenta contains (para casos como "CD LITORAL" vs "LITORAL")
        if not mask.any():
            mask = df[coluna_encontrada].astype(str).str.contains(unidade, case=False, na=False)
        
        df_filtrado = df[mask].copy()
        
        # Mostra no log quantos registros foram filtrados (opcional)
        if len(df_filtrado) < len(df):
            st.sidebar.caption(f"📊 {coluna_encontrada}: {len(df_filtrado)} registros de {len(df)}")
        
        return df_filtrado
    
    # Se não encontrou nenhuma coluna, retorna o DataFrame original (com aviso)
    st.sidebar.warning(f"⚠️ Não foi possível filtrar por unidade. Coluna não encontrada.")
    return df

# sistema_login.py
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
    """Carrega as credenciais do arquivo Excel"""
    # Caminho relativo - ajuste conforme sua estrutura
    caminho_credenciais = Path(__file__).resolve().parent / "data" / "credenciais.xlsx"
    
    # Tentar outros caminhos comuns
    if not caminho_credenciais.exists():
        caminhos_alternativos = [
            Path("data/credenciais.xlsx"),
            Path("../data/credenciais.xlsx"),
            Path("./data/credenciais.xlsx"),
        ]
        for alt in caminhos_alternativos:
            if alt.exists():
                caminho_credenciais = alt
                break
    
    if not caminho_credenciais.exists():
        # Fallback para desenvolvimento (NÃO USAR EM PRODUÇÃO)
        st.warning("Arquivo de credenciais não encontrado. Usando dados padrão.")
        return pd.DataFrame([
            {"Usuario": "Adm", "Senha": hash_senha("Adm@log20@"), "Unidade": "Todas"},
            {"Usuario": "cd.litoral", "Senha": hash_senha("lito@log20"), "Unidade": "CD LITORAL"},
            {"Usuario": "cd.petrop", "Senha": hash_senha("petr@log20"), "Unidade": "CD PETRÓPOLIS"},
        ])
    
    # Carregar e preparar credenciais
    credenciais = pd.read_excel(caminho_credenciais)
    
    # Garantir que as colunas existem
    if "Senha" in credenciais.columns and "Senha" not in credenciais.columns.str.upper():
        # Se a senha não está hasheada, fazer o hash
        if not credenciais["Senha"].astype(str).str.startswith("hash_").any():
            credenciais["Senha_Hash"] = credenciais["Senha"].apply(hash_senha)
            credenciais["Senha"] = credenciais["Senha_Hash"]
    
    return credenciais

def fazer_login():
    """Exibe formulário de login e verifica credenciais"""
    
    # Verificar se já está logado e sessão ainda é válida
    if "logado" in st.session_state and st.session_state.logado:
        # Opcional: expirar sessão após 8 horas
        if "login_time" in st.session_state:
            tempo_logado = (datetime.now() - st.session_state.login_time).total_seconds() / 3600
            if tempo_logado > 8:
                logout()
                return fazer_login()
        return True
    
    # CSS para tela de login moderna
    st.markdown("""
    <style>
    .login-container {
        max-width: 420px;
        margin: 80px auto;
        padding: 35px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        border: 1px solid rgba(240, 211, 107, 0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
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
    .login-error {
        background: rgba(220, 53, 69, 0.2);
        color: #ff6b6b;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    .login-footer {
        text-align: center;
        margin-top: 20px;
        color: #666;
        font-size: 12px;
    }
    div[data-testid="stForm"] {
        background: transparent;
        border: none;
        padding: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Container centralizado
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown('<div class="login-title">📦 Dashboard Integração</div>', unsafe_allow_html=True)
            st.markdown('<div class="login-subtitle">Sistema de Gestão de Padrinhos e Integração</div>', unsafe_allow_html=True)
            
            # Formulário de login
            with st.form("login_form"):
                usuario = st.text_input("👤 Usuário", placeholder="Digite seu usuário (ex: cd.litoral)")
                senha = st.text_input("🔒 Senha", type="password", placeholder="Digite sua senha")
                submitted = st.form_submit_button("🔐 Entrar", use_container_width=True)
                
                if submitted:
                    if not usuario or not senha:
                        st.markdown('<div class="login-error">❌ Preencha usuário e senha</div>', unsafe_allow_html=True)
                        return False
                    
                    credenciais = carregar_credenciais()
                    senha_hash = hash_senha(senha)
                    
                    # Buscar usuário (comparar com hash)
                    # Verificar tanto coluna "Senha" (hash) quanto "Senha_Hash"
                    col_senha = None
                    if "Senha_Hash" in credenciais.columns:
                        col_senha = "Senha_Hash"
                    elif "Senha" in credenciais.columns:
                        col_senha = "Senha"
                    
                    if col_senha:
                        user_data = credenciais[
                            (credenciais["Usuario"].astype(str).str.lower() == usuario.lower()) & 
                            (credenciais[col_senha] == senha_hash)
                        ]
                    else:
                        user_data = pd.DataFrame()
                    
                    if not user_data.empty:
                        # Login bem-sucedido
                        st.session_state.logado = True
                        st.session_state.usuario = user_data.iloc[0]["Usuario"]
                        st.session_state.unidade = user_data.iloc[0]["Unidade"]
                        st.session_state.login_time = datetime.now()
                        
                        st.success(f"✅ Bem-vindo, {usuario}!")
                        st.balloons()
                        st.rerun()
                        return True
                    else:
                        st.markdown('<div class="login-error">❌ Usuário ou senha inválidos</div>', unsafe_allow_html=True)
                        return False
            
            st.markdown('<div class="login-footer">🔒 Ambiente seguro - Acesso restrito aos gestores</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    return False

def logout():
    """Realiza logout do sistema"""
    for key in ["logado", "usuario", "unidade", "permissao", "login_time"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def exibir_info_usuario():
    """Exibe informações do usuário logado no sidebar"""
    if st.session_state.get("logado", False):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"""
        <div style="background: #1e1e2e; padding: 12px; border-radius: 10px; margin: 10px 0;">
            <div style="color: #f0d36b; font-weight: bold; margin-bottom: 8px;">👤 Usuário Logado</div>
            <div style="font-size: 13px;">📛 <strong>Usuário:</strong> {st.session_state.usuario}</div>
            <div style="font-size: 13px;">🏭 <strong>Unidade:</strong> {st.session_state.unidade}</div>
            <div style="font-size: 12px; color: #888; margin-top: 8px;">🕐 Login: {st.session_state.login_time.strftime('%d/%m/%Y %H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("🚪 Sair do Sistema", use_container_width=True):
            logout()

def aplicar_filtro_unidade(df: pd.DataFrame, coluna_unidade: str = "Operação") -> pd.DataFrame:
    """Aplica filtro de unidade baseado no usuário logado"""
    if not st.session_state.get("logado", False):
        return df
    
    unidade_usuario = st.session_state.unidade
    
    # Admin vê todas as unidades
    if unidade_usuario == "Todas" or st.session_state.usuario == "Adm":
        return df
    
    if df.empty:
        return df
    
    # Tentar encontrar a coluna de operação/unidade
    coluna_encontrada = None
    possiveis_nomes = [coluna_unidade, "Operação", "Unidade", "CD", "Filial"]
    
    for nome in possiveis_nomes:
        for col in df.columns:
            if nome.lower() in col.lower():
                coluna_encontrada = col
                break
        if coluna_encontrada:
            break
    
    if coluna_encontrada:
        # Filtro exato
        df_filtrado = df[df[coluna_encontrada].astype(str).str.strip() == unidade_usuario].copy()
        
        # Se não encontrar exato, tenta contém
        if df_filtrado.empty:
            df_filtrado = df[df[coluna_encontrada].astype(str).str.contains(unidade_usuario, case=False, na=False)].copy()
        
        return df_filtrado
    
    return df

def verificar_permissao(permissao_necessaria: str = "visualizador") -> bool:
    """Verifica se o usuário tem permissão para acessar determinada funcionalidade"""
    if not st.session_state.get("logado", False):
        return False
    
    # Admin tem todas as permissões
    if st.session_state.usuario == "Adm":
        return True
    
    # Mapeamento de permissões
    permissoes = {
        "visualizador": ["visualizador"],
        "gestor": ["visualizador", "gestor"],
        "admin": ["visualizador", "gestor", "admin"]
    }
    
    permissao_usuario = st.session_state.get("permissao", "visualizador")
    
    return permissao_necessaria in permissoes.get(permissao_usuario, ["visualizador"])

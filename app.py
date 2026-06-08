# app.py - VERSÃO CORRIGIDA
import streamlit as st
import pandas as pd
from pathlib import Path

# =========================
# FUNÇÕES DE LOGIN
# =========================

@st.cache_data(ttl=3600)
def carregar_credenciais():
    """Carrega a planilha de credenciais"""
    caminhos = [
        Path("data/credenciais.xlsx"),
        Path("credenciais.xlsx"),
        Path("../credenciais.xlsx"),
    ]
    for caminho in caminhos:
        if caminho.exists():
            return pd.read_excel(caminho)
    
    # Se não encontrar, cria uma padrão
    st.warning("Arquivo de credenciais não encontrado. Criando padrão...")
    df = pd.DataFrame([
        {"Usuario": "Adm", "Senha": "Adm@log20@", "Operação": "Todas"},
        {"Usuario": "cd.litoral", "Senha": "lito@log20", "Operação": "CD LITORAL"},
        {"Usuario": "cd.petrop", "Senha": "petr@log20", "Operação": "CD PETRÓPOLIS"},
        {"Usuario": "cd.cascave", "Senha": "casc@log20", "Operação": "CD CASCAVEL"},
    ])
    
    # Tentar salvar
    try:
        Path("data").mkdir(exist_ok=True)
        df.to_excel("data/credenciais.xlsx", index=False)
        st.success("Arquivo de credenciais criado em data/credenciais.xlsx")
    except:
        pass
    
    return df

def fazer_login():
    """Verifica login sem hash (compara texto diretamente)"""
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
    st.markdown("Digite suas credenciais")
    
    with st.form("login_form"):
        usuario = st.text_input("👤 Usuário")
        senha = st.text_input("🔒 Senha", type="password")
        submitted = st.form_submit_button("📥 Entrar", use_container_width=True)
        
        if submitted:
            if not usuario or not senha:
                st.error("❌ Preencha usuário e senha")
                return False
            
            creds = carregar_credenciais()
            
            # Comparação direta (sem hash)
            user = creds[
                (creds["Usuario"].astype(str).str.strip() == usuario) & 
                (creds["Senha"].astype(str).str.strip() == senha)
            ]
            
            if not user.empty:
                st.session_state.logado = True
                st.session_state.usuario = user.iloc[0]["Usuario"]
                st.session_state.operacao = user.iloc[0]["Operação"]  # ← com acento
                st.success(f"✅ Bem-vindo, {usuario}!")
                st.rerun()
                return True
            else:
                st.error("❌ Usuário ou senha inválidos")
                with st.expander("🔧 Debug - Usuários disponíveis"):
                    st.dataframe(creds[["Usuario", "Operação"]])
    
    st.markdown('</div>', unsafe_allow_html=True)
    return False

def get_operacao():
    """Retorna a operação do usuário logado"""
    return st.session_state.get("operacao", "Todas")  # ← 'operacao' minúsculo

def get_usuario():
    """Retorna o nome do usuário logado"""
    return st.session_state.get("usuario", "")

def aplicar_filtro(df, coluna="Operação"):
    """Filtra DataFrame pela operação do usuário"""
    if df is None or df.empty:
        return df
    
    operacao = get_operacao()
    
    # Admin vê tudo
    if operacao == "Todas" or get_usuario() == "Adm":
        return df
    
    # Verifica se a coluna existe
    if coluna not in df.columns:
        st.warning(f"⚠️ Coluna '{coluna}' não encontrada. Colunas: {list(df.columns)}")
        return df
    
    # Aplica o filtro
    df_filtrado = df[df[coluna].astype(str).str.strip() == operacao].copy()
    
    return df_filtrado

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

# Botões para cada dashboard
col1, col2 = st.columns(2)

with col1:
    if st.button("🔐 Painel de Acessos RH", use_container_width=True):
        st.switch_page("pages/1_Painel de Acessos RH.py")
    
    if st.button("📚 Materiais Integrações", use_container_width=True):
        st.switch_page("pages/2_Materiais_integracoes.py")
    
    if st.button("🚚 Integração Distribuição", use_container_width=True):
        st.switch_page("pages/3_Integracao_Distribuicao.py")
    
    if st.button("📦 Integração Armazém", use_container_width=True):
        st.switch_page("pages/4_Integracao_Armazem.py")

with col2:
    if st.button("👨🏻‍🎓 Padrinhos", use_container_width=True):
        st.switch_page("pages/4_Padrinhos.py")
    
    if st.button("📋 Acompanhamento de Novos", use_container_width=True):
        st.switch_page("pages/Acompanhamento_de_Novos.py")
    
    if st.button("📈 Raio X", use_container_width=True):
        st.switch_page("pages/Raio X Operação.py")
    
    if st.button("📊 SKAP", use_container_width=True):
        st.switch_page("pages/SKAP.py")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
import plotly.express as px
import plotly.graph_objects as go
import locale
import os
from io import BytesIO

# Tentar configurar locale para português
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'portuguese')
    except:
        pass

# =========================
# Configuração da página
# =========================
st.set_page_config(
    page_title="Dashboard Integração Armazém", 
    layout="wide",
    page_icon="📦",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] {
        background-color: #2c2c2c !important;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetric"] label {
        color: white !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetric"] div {
        color: white !important;
    }
    .info-colaborador {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #1f77b4;
    }
    .pergunta-texto {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 10px;
        word-wrap: break-word;
        white-space: normal;
    }
    .grafico-titulo {
        background-color: #2c2c2c;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        font-weight: bold;
        text-align: center;
    }
    .recarregar-btn {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Funções auxiliares
# =========================

def similar(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

# ===========================================
# VERSÃO OTIMIZADA - FUZZY MATCH VETORIZADO
# ===========================================
def fuzzy_match_vectorized(df_colaboradores, df_mod):
    """
    Versão otimizada do fuzzy match usando vetorização parcial
    """
    if len(df_colaboradores) == 0 or len(df_mod) == 0:
        return []
    
    resultados = []
    
    # Pré-calcular nomes e CPFs normalizados uma única vez
    df_colaboradores['_nome_norm'] = df_colaboradores['Colaborador'].astype(str).str.lower().str.strip()
    df_colaboradores['_cpf_norm'] = df_colaboradores['CPF'].astype(str).str.lower().str.strip()
    
    df_mod['_nome_norm'] = df_mod['Colaborador'].astype(str).str.lower().str.strip()
    df_mod['_cpf_norm'] = df_mod['CPF'].astype(str).str.lower().str.strip()
    
    # Criar lista para busca rápida
    mod_dict = []
    for _, tgt in df_mod.iterrows():
        mod_dict.append({
            'nome': tgt['_nome_norm'],
            'cpf': tgt['_cpf_norm'],
            'original': tgt
        })
    
    for _, row in df_colaboradores.iterrows():
        matched = None
        best_score = 0
        
        for tgt in mod_dict:
            # Score do nome
            score_name = similar(row['_nome_norm'], tgt['nome'])
            score_cpf = similar(row['_cpf_norm'], tgt['cpf'])
            final_score = max(score_name, score_cpf)
            
            if final_score > best_score:
                best_score = final_score
                if final_score >= 0.65:
                    matched = tgt['original']
                else:
                    matched = None
        
        resultados.append(matched)
    
    return resultados

def calcular_estatisticas_respostas(df_mod, perguntas_gabarito, respostas_validas):
    """Calcula estatísticas de acertos por pergunta - OTIMIZADO"""
    estatisticas = []
    
    colunas_df = [col for col in df_mod.columns if col not in ['Colaborador', 'CPF']]
    
    for idx, (pergunta_gabarito, resposta_correta) in enumerate(zip(perguntas_gabarito, respostas_validas)):
        coluna_encontrada = None
        melhor_similaridade = 0
        
        for coluna_df in colunas_df:
            similaridade = similar(pergunta_gabarito, coluna_df)
            if similaridade > melhor_similaridade and similaridade > 0.6:
                melhor_similaridade = similaridade
                coluna_encontrada = coluna_df
        
        if coluna_encontrada is None:
            if idx < len(colunas_df):
                coluna_encontrada = colunas_df[idx]
            else:
                continue
        
        # VETORIZADO: cálculo de acertos usando pandas diretamente
        if coluna_encontrada in df_mod.columns:
            series_resposta = df_mod[coluna_encontrada].astype(str).str.strip().str.lower()
            resposta_correta_norm = resposta_correta.strip().lower()
            
            # Filtra respostas válidas (não vazias e não 'nan')
            mask_valida = (series_resposta != '') & (series_resposta != 'nan')
            total = mask_valida.sum()
            
            if total > 0:
                acertos = (series_resposta[mask_valida] == resposta_correta_norm).sum()
                percentual_acerto = (acertos / total) * 100
                percentual_erro = 100 - percentual_acerto
            else:
                percentual_acerto = 0
                percentual_erro = 0
        else:
            percentual_acerto = 0
            percentual_erro = 0
            total = 0
            
        estatisticas.append({
            'Pergunta': pergunta_gabarito,
            'Acertos (%)': round(percentual_acerto, 2),
            'Erros (%)': round(percentual_erro, 2),
            'Total_Respostas': total,
            'Coluna_Utilizada': coluna_encontrada
        })
    
    return estatisticas

def gerar_excel_detalhamento(df_detalhamento):
    """Gera arquivo Excel para download do detalhamento"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_detalhamento.to_excel(writer, index=False, sheet_name='Detalhamento')
    return output.getvalue()

# =========================
# Cache de dados com botão de recarregar
# =========================

@st.cache_data(ttl=3600)
def load_data():
    """Carrega os dados das planilhas"""
    possiveis_caminhos = [
        "data/Base_Colaboradores_Ativos.xlsx",
        "Base_Colaboradores_Ativos.xlsx",
        "../data/Base_Colaboradores_Ativos.xlsx",
        "./data/Base_Colaboradores_Ativos.xlsx"
    ]
    
    admitidos = pd.read_excel("data/Admitidos.xlsx")
    integracao = {m: pd.read_excel("data/Integracao Armazem.xlsx", sheet_name=m) for m in ["M1", "M2", "M3", "M4", "M5"]}
    
    colaboradores_ativos_lista = []
    arquivo_encontrado = False
    caminho_utilizado = None
    
    for caminho in possiveis_caminhos:
        if os.path.exists(caminho):
            try:
                colaboradores_ativos = pd.read_excel(caminho)
                if 'Colaborador' in colaboradores_ativos.columns:
                    colaboradores_ativos_lista = colaboradores_ativos['Colaborador'].tolist()
                    arquivo_encontrado = True
                    caminho_utilizado = caminho
                    break
                else:
                    st.warning(f"⚠️ Arquivo {caminho} não tem a coluna 'Colaborador'")
            except Exception as e:
                st.warning(f"⚠️ Erro ao ler {caminho}: {str(e)}")
    
    if not arquivo_encontrado:
        colaboradores_ativos_lista = admitidos['Colaborador'].tolist()
    
    return admitidos, integracao, colaboradores_ativos_lista, caminho_utilizado

def recarregar_cache():
    st.cache_data.clear()
    st.success("✅ Cache recarregado com sucesso!")
    st.rerun()

# =========================
# Carregamento dos dados
# =========================

with st.spinner('Carregando dados...'):
    admitidos, integracao, colaboradores_ativos_lista, caminho_base_ativos = load_data()
    
    if caminho_base_ativos:
        st.sidebar.success(f"✅ Base ativos: {len(colaboradores_ativos_lista)} colaboradores")
    else:
        st.sidebar.warning("⚠️ Usando fallback: todos colaboradores como ativos")

# =========================
# Processamento inicial (OTIMIZADO - VETORIZADO)
# =========================
admitidos = admitidos[admitidos['Cargo'] == "Ajudante Armazém"]

# Filtro vetorizado para excluir operações
operacoes_excluir = ["VIDROS PR", "PONTA GROSSA"]
admitidos = admitidos[~admitidos['Operação'].isin(operacoes_excluir)]

admitidos['Data'] = pd.to_datetime(admitidos['Data'], errors='coerce')

# Status vetorizado (usando isin)
colaboradores_ativos_set = set(colaboradores_ativos_lista)
admitidos['Status'] = admitidos['Colaborador'].apply(lambda x: 'Ativo' if x in colaboradores_ativos_set else 'Inativo')

# Filtrar apenas ativos
admitidos = admitidos[admitidos['Status'] == 'Ativo'].copy()

# Regras específicas por unidade (vetorizado)
data_corte_petropolis = datetime(2025, 10, 1)
data_corte_litoral = datetime(2024, 11, 1)

mask_petropolis = (admitidos['Operação'] == "CD PETRÓPOLIS") & (admitidos['Data'] < data_corte_petropolis)
mask_litoral = (admitidos['Operação'] == "CD LITORAL") & (admitidos['Data'] < data_corte_litoral)
admitidos = admitidos[~(mask_petropolis | mask_litoral)]

hoje_corte = datetime(2026, 2, 28)
if datetime.now() <= hoje_corte:
    admitidos = admitidos[admitidos['Turno'].str.upper() == "NOITE"]

# =========================
# Sidebar - Filtros
# =========================
with st.sidebar:
    st.markdown("## 🎛️ Filtros")
    
    operacoes = sorted(admitidos['Operação'].unique())
    filtro_operacao = st.selectbox(
        "🏭 Operação", 
        ["Todas"] + operacoes,
        help="Selecione uma operação específica"
    )
    
    min_date = admitidos['Data'].min().date()
    max_date = admitidos['Data'].max().date()
    data_padrao_inicio = datetime(2025, 5, 1).date()
    
    if data_padrao_inicio < min_date:
        data_padrao_inicio = min_date
    elif data_padrao_inicio > max_date:
        data_padrao_inicio = max_date
    
    col1, col2 = st.columns(2)
    with col1:
        data_inicio = st.date_input(
            "📅 Data inicial", 
            value=data_padrao_inicio,
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY"
        )
    with col2:
        data_fim = st.date_input(
            "📅 Data final", 
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY"
        )
    
    colaboradores = sorted(admitidos['Colaborador'].unique())
    filtro_colaborador = st.selectbox(
        "👤 Colaborador", 
        ["Todos"] + colaboradores,
        help="Selecione um colaborador específico"
    )
    
    filtro_status_modulo = st.multiselect(
        "📋 Status do Módulo",
        options=["Realizado", "Não realizado"],
        default=["Realizado", "Não realizado"],
        help="Filtrar por status de realização do módulo"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Informações")
    st.caption(f"📅 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption(f"👥 Mostrando apenas colaboradores ATIVOS")
    
    st.markdown("---")
    
    st.markdown("### 🔄 Controle de Dados")
    if st.button("🔄 Recarregar Cache (uso corporativo)", use_container_width=True):
        recarregar_cache()

# =========================
# Aplicando filtros
# =========================
admitidos_filtrado = admitidos.copy()

if filtro_operacao != "Todas":
    admitidos_filtrado = admitidos_filtrado[admitidos_filtrado['Operação'] == filtro_operacao]

admitidos_filtrado = admitidos_filtrado[
    (admitidos_filtrado['Data'] >= pd.to_datetime(data_inicio)) & 
    (admitidos_filtrado['Data'] <= pd.to_datetime(data_fim))
]

colaborador_selecionado = None
if filtro_colaborador != "Todos":
    admitidos_filtrado = admitidos_filtrado[
        admitidos_filtrado['Colaborador'] == filtro_colaborador
    ]
    colaborador_selecionado = filtro_colaborador

# =========================
# Fuzzy Match (VERSÃO OTIMIZADA)
# =========================
results_list = []

for modulo, df_mod in integracao.items():
    # Verificar se as colunas necessárias existem
    if 'Colaborador' not in df_mod.columns:
        df_mod['Colaborador'] = ''
    if 'CPF' not in df_mod.columns:
        df_mod['CPF'] = ''
    
    # Usar função otimizada
    matches = fuzzy_match_vectorized(admitidos_filtrado, df_mod)
    
    for idx, row in admitidos_filtrado.iterrows():
        matched = matches[idx]
        status = "Realizado" if matched is not None else "Não realizado"
        results_list.append({
            "Operação": row['Operação'],
            "Colaborador": row['Colaborador'],
            "Data_Admissao": row['Data'],
            "Módulo": modulo,
            "Status_Modulo": status
        })

resultado_modulos = pd.DataFrame(results_list)

if filtro_status_modulo and len(resultado_modulos) > 0:
    resultado_modulos = resultado_modulos[resultado_modulos['Status_Modulo'].isin(filtro_status_modulo)]

# =========================
# Título e gráfico de aderência geral
# =========================
st.title("📦 Dashboard de Integração Armazém")

if len(resultado_modulos) > 0:
    aderencia_geral = resultado_modulos.groupby('Operação')['Status_Modulo'].apply(
        lambda x: (x == "Realizado").mean() * 100
    ).reset_index()
    aderencia_geral.columns = ['Operação', 'Aderência (%)']
    
    aderencia_geral = aderencia_geral.sort_values('Aderência (%)', ascending=False)
    
    fig_aderencia = px.bar(
        aderencia_geral,
        x='Operação',
        y='Aderência (%)',
        title='Aderência Geral por Operação',
        text='Aderência (%)',
        color='Aderência (%)',
        color_continuous_scale=['red', 'yellow', 'green'],
        range_color=[0, 100]
    )
    fig_aderencia.update_traces(
        texttemplate='%{text:.1f}%', 
        textposition='outside',
        textfont=dict(size=12, color='black')
    )
    fig_aderencia.update_layout(
        xaxis_title="Operação",
        yaxis_title="Aderência (%)",
        yaxis_range=[0, 100],
        height=500,
        showlegend=False,
        uniformtext_minsize=10,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig_aderencia, use_container_width=True, key="aderencia_geral")
else:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")

st.markdown("---")

# =========================
# Tabela detalhada
# =========================
st.subheader("📄 Detalhamento por Colaborador")

if len(resultado_modulos) > 0:
    modulos_disponiveis = ["Módulo 1", "Módulo 2", "Módulo 3", "Módulo 4", "Módulo 5"]
    modulo_map = {"Módulo 1": "M1", "Módulo 2": "M2", "Módulo 3": "M3", "Módulo 4": "M4", "Módulo 5": "M5"}
    
    filtro_modulo_detalhe = st.selectbox(
        "Filtrar por módulo",
        options=["Todos"] + modulos_disponiveis,
        index=0,
        key="filtro_modulo"
    )
    
    if filtro_modulo_detalhe == "Todos":
        tabela_filtrada = resultado_modulos.copy()
    else:
        modulo_selecionado = modulo_map[filtro_modulo_detalhe]
        tabela_filtrada = resultado_modulos[resultado_modulos['Módulo'] == modulo_selecionado]
    
    tabela_detalhada = tabela_filtrada[['Operação', 'Colaborador', 'Data_Admissao', 'Módulo', 'Status_Modulo']].copy()
    
    modulo_nomes = {"M1": "Módulo 1", "M2": "Módulo 2", "M3": "Módulo 3", "M4": "Módulo 4", "M5": "Módulo 5"}
    tabela_detalhada['Módulo'] = tabela_detalhada['Módulo'].map(modulo_nomes)
    
    tabela_detalhada['Data_Admissao'] = pd.to_datetime(tabela_detalhada['Data_Admissao']).dt.strftime('%d/%m/%Y')
    
    tabela_detalhada.columns = ['Operação', 'Colaborador', 'Admissão', 'Módulo', 'Status do Módulo']
    
    tabela_detalhada = tabela_detalhada.drop_duplicates(subset=['Operação', 'Colaborador', 'Módulo'])
    tabela_detalhada = tabela_detalhada.sort_values(['Operação', 'Colaborador', 'Módulo'])
    
    def color_status(val):
        if val == 'Realizado':
            return 'background-color: #28a745; color: white'
        else:
            return 'background-color: #dc3545; color: white'
    
    styled_table = tabela_detalhada.style.map(color_status, subset=['Status do Módulo'])
    
    st.dataframe(styled_table, use_container_width=True, height=400)
    
    excel_data = gerar_excel_detalhamento(tabela_detalhada)
    st.download_button(
        label="📥 Baixar Detalhamento em Excel",
        data=excel_data,
        file_name=f"detalhamento_integracao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
else:
    st.info("Nenhum dado para exibir na tabela detalhada.")

st.markdown("---")

# =========================
# Análise de Respostas Check de Retenção
# =========================
st.header("📝 Análise Respostas Check de Retenção")

if colaborador_selecionado:
    dados_colaborador = admitidos_filtrado[admitidos_filtrado['Colaborador'] == colaborador_selecionado]
    if not dados_colaborador.empty:
        data_admissao = dados_colaborador.iloc[0]['Data'].strftime('%d/%m/%Y')
        operacao_colaborador = dados_colaborador.iloc[0]['Operação']
        
        st.markdown(f"""
        <div class="info-colaborador">
            <strong>👤 Colaborador:</strong> {colaborador_selecionado}<br>
            <strong>📅 Data de Admissão:</strong> {data_admissao}<br>
            <strong>🏭 Operação:</strong> {operacao_colaborador}<br>
            <strong>📌 Status:</strong> Ativo
        </div>
        """, unsafe_allow_html=True)

# Gabaritos
gabaritos = {
    "M1": {
        "nome": "Módulo 1",
        "pontuacao": 90,
        "perguntas": [
            "Selecione o EPI que NÃO é obrigatório durante o processo de montagem de paletes",
            "Selecione a opção CORRETA sobre o manuseio de barris de chopp",
            "Selecione a opção sobre o procedimento CORRETO",
            "Antes de iniciar o processo de montagem, DEVE-SE:",
            "Sobre o manuseio de Market Place, selecione a opção CORRETA",
            "Selecione a opção CORRETA sobre a organização do local",
            "De que forma conseguimos garantir a execução do FEFO?",
            "Sobre a operação do repack, selecione a opção INCORRETA:",
            "Sobre a operação de descarga de veículos, selecione a opção CORRETA:"
        ],
        "respostas": [
            "Cinto paraquedista anti queda",
            "Todo barril deve ser movimentado por duas pessoas utilizando os EPIs corretos",
            "Durante as atividades do Picking, deve-se manter a segregação homem x máquina",
            "Validar a condição do palete e, em caso de más condições, o palete de madeira deve ser segregado e substituído",
            "Devemos realizar a movimentação de garrafas de forma individual e com as duas mãos",
            "Deve-se garantir que o local está limpo e sem a presença de nenhum obstáculo que atrapalhe a movimentação dos produtos",
            "Os produtos com data curta devem estar disponíveis antes das datas longas, desde que estejam dentro do prazo comercial",
            "Podemos realizar a reembalagem de produtos com diferentes datas de validade",
            "A chave do caminhão nunca pode estar na ignição"
        ]
    },
    "M2": {
        "nome": "Módulo 2",
        "pontuacao": 40,
        "perguntas": [
            "Selecione a melhor alternativa que resuma a curva ABC",
            "Selecione a opção CORRETA sobre ordem correta de paletização (OCP)",
            "Selecione a opção que MELHOR caracteriza o Picking",
            "Selecione a alternativa CORRETA sobre o layout do Picking"
        ],
        "respostas": [
            "Curva A: \"Alto giro\"; Curva B: \"Médio giro\"; Curva C: \"Baixo giro\"",
            "Todas as alternativas estão corretas",
            "O Picking é a área que é projetada para realizarmos a montagem dos paletes durante a separação",
            "O layout ideal do Picking é organizar as posições baseadas por embalagem"
        ]
    },
    "M3": {
        "nome": "Módulo 3",
        "pontuacao": 40,
        "perguntas": [
            "Selecione a alternativa CORRETA sobre montagem de paletes",
            "Selecione a opção que considera os principais elementos de um palete",
            "Selecione a opção CORRETA sobre montagem de lastros",
            "Selecione a opção CORRETA sobre o processo de montagem"
        ],
        "respostas": [
            "Os produtos mistos devem ser colocados nas laterais do palete para facilitar a conferência",
            "Palete de madeira, lastro e elemento divisório (chapatex e folha separadora)",
            "Os produtos diversos precisam ser sempre colocados na parte mais externa do lastro para que estejam visíveis durante a conferência",
            "Após a separação do palete, o mesmo é conferido pelo conferente e, em seguida, liberado para carregamento"
        ]
    },
    "M4": {
        "nome": "Módulo 4",
        "pontuacao": 40,
        "perguntas": [
            "Selecione a opção CORRETA sobre o WMS",
            "Selecione a opção INCORRETA sobre o módulo de Separação do WMS",
            "Selecione a opção CORRETA sobre o processo de separação no WMS",
            "Selecione as etapas CORRETAS do processo de separação"
        ],
        "respostas": [
            "Caso tenha problemas com meu usuário devo utilizar a função \"Esqueci a senha\" ou procurar suporte do conferente",
            "Não é mostrada a quantidade correta do produto na tela do WMS",
            "Devemos ter atenção na descrição por completo",
            "Associar palete, visualizar produto, realizar separação, conferir e finalizar palete"
        ]
    },
    "M5": {
        "nome": "Módulo 5",
        "pontuacao": 40,
        "perguntas": [
            "Selecione a opção CORRETA sobre pontuação",
            "Selecione a opção CORRETA sobre remuneração variável",
            "Selecione os PRINCIPAIS indicadores de produtividade",
            "Selecione a opção que NÃO se trata de erro de montagem"
        ],
        "respostas": [
            "Há uma memória de cálculo para classificar a complexidade dos paletes",
            "Quanto mais montar, maior a remuneração variável",
            "Ponto laboral, PPS (Ponto por segundo) e EFM (Eficiência de Montagem)",
            "Separar a quantidade correta, do produto correto e em boas condições"
        ]
    }
}

modulos_lista = list(integracao.keys())
tabs = st.tabs([f"📘 {gabaritos[modulo]['nome']}" for modulo in modulos_lista])

for idx, modulo in enumerate(modulos_lista):
    with tabs[idx]:
        st.subheader(f"Análise de Respostas - {gabaritos[modulo]['nome']}")
        
        df_mod = integracao[modulo]
        
        if modulo not in gabaritos:
            st.warning(f"Gabarito não encontrado para o módulo {modulo}")
            continue
            
        respostas_validas = gabaritos[modulo]['respostas']
        perguntas_gabarito = gabaritos[modulo]['perguntas']
        
        estatisticas = calcular_estatisticas_respostas(df_mod, perguntas_gabarito, respostas_validas)
        
        for i in range(0, len(estatisticas), 2):
            cols = st.columns(2)
            
            if i < len(estatisticas):
                with cols[0]:
                    st.markdown(f'<div class="grafico-titulo">📊 Pergunta {i+1}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="pergunta-texto">{i+1}. {estatisticas[i]["Pergunta"]}</div>', unsafe_allow_html=True)
                    
                    df_pergunta = pd.DataFrame({
                        'Status': ['Acertos', 'Erros'],
                        'Percentual': [estatisticas[i]['Acertos (%)'], estatisticas[i]['Erros (%)']]
                    })
                    
                    fig = px.bar(
                        df_pergunta,
                        x='Status',
                        y='Percentual',
                        text='Percentual',
                        color='Status',
                        color_discrete_map={'Acertos': '#28a745', 'Erros': '#dc3545'},
                        labels={'Percentual': 'Percentual (%)', 'Status': ''}
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        yaxis_range=[0, 100],
                        height=350,
                        showlegend=False,
                        xaxis_title="",
                        yaxis_title="Percentual (%)"
                    )
                    
                    chart_key = f"chart_{modulo}_{i}_{datetime.now().timestamp()}"
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("✅ Acertos", f"{estatisticas[i]['Acertos (%)']:.1f}%")
                    with col2:
                        st.metric("❌ Erros", f"{estatisticas[i]['Erros (%)']:.1f}%")
                    
                    st.caption(f"📊 Total: {estatisticas[i]['Total_Respostas']} respostas")
                    st.markdown("---")
            
            if i + 1 < len(estatisticas):
                with cols[1]:
                    st.markdown(f'<div class="grafico-titulo">📊 Pergunta {i+2}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="pergunta-texto">{i+2}. {estatisticas[i+1]["Pergunta"]}</div>', unsafe_allow_html=True)
                    
                    df_pergunta = pd.DataFrame({
                        'Status': ['Acertos', 'Erros'],
                        'Percentual': [estatisticas[i+1]['Acertos (%)'], estatisticas[i+1]['Erros (%)']]
                    })
                    
                    fig = px.bar(
                        df_pergunta,
                        x='Status',
                        y='Percentual',
                        text='Percentual',
                        color='Status',
                        color_discrete_map={'Acertos': '#28a745', 'Erros': '#dc3545'},
                        labels={'Percentual': 'Percentual (%)', 'Status': ''}
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        yaxis_range=[0, 100],
                        height=350,
                        showlegend=False,
                        xaxis_title="",
                        yaxis_title="Percentual (%)"
                    )
                    
                    chart_key = f"chart_{modulo}_{i+1}_{datetime.now().timestamp()}"
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("✅ Acertos", f"{estatisticas[i+1]['Acertos (%)']:.1f}%")
                    with col2:
                        st.metric("❌ Erros", f"{estatisticas[i+1]['Erros (%)']:.1f}%")
                    
                    st.caption(f"📊 Total: {estatisticas[i+1]['Total_Respostas']} respostas")
                    st.markdown("---")

st.markdown("### 📌 Legenda")
col_leg1, col_leg2, col_leg3 = st.columns(3)
with col_leg1:
    st.markdown("🟩 **Acertos** - Respostas corretas")
with col_leg2:
    st.markdown("🟥 **Erros** - Respostas incorretas")
with col_leg3:
    st.markdown("📊 **Total** - Quantidade de respostas analisadas")

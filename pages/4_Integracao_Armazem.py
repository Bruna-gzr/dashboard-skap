import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
import plotly.express as px
import plotly.graph_objects as go

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
</style>
""", unsafe_allow_html=True)

# =========================
# Funções auxiliares
# =========================

def similar(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def fuzzy_match(row, df_target):
    best_score = 0
    best_row = None

    for _, tgt in df_target.iterrows():
        score_name = similar(str(row['Colaborador']), str(tgt['Colaborador']))
        score_cpf = similar(str(row['CPF']), str(tgt['CPF']))
        final_score = max(score_name, score_cpf)
        if final_score > best_score:
            best_score = final_score
            best_row = tgt

    return best_row if best_score >= 0.65 else None

def calcular_estatisticas_respostas(df_mod, perguntas_gabarito, respostas_validas):
    """Calcula estatísticas de acertos por pergunta"""
    estatisticas = []
    
    # Obter as colunas do DataFrame (removendo Colaborador e CPF)
    colunas_df = [col for col in df_mod.columns if col not in ['Colaborador', 'CPF']]
    
    # Para cada pergunta do gabarito, encontrar a coluna correspondente no DataFrame
    for idx, (pergunta_gabarito, resposta_correta) in enumerate(zip(perguntas_gabarito, respostas_validas)):
        # Tentar encontrar a coluna que mais se parece com a pergunta do gabarito
        coluna_encontrada = None
        melhor_similaridade = 0
        
        for coluna_df in colunas_df:
            similaridade = similar(pergunta_gabarito, coluna_df)
            if similaridade > melhor_similaridade and similaridade > 0.6:
                melhor_similaridade = similaridade
                coluna_encontrada = coluna_df
        
        if coluna_encontrada is None:
            # Se não encontrar, usar a primeira coluna disponível (fallback)
            if idx < len(colunas_df):
                coluna_encontrada = colunas_df[idx]
            else:
                continue
        
        # Calcular acertos e erros
        acertos = 0
        total = 0
        
        for _, row in df_mod.iterrows():
            resposta_usuario = str(row[coluna_encontrada]).strip().lower()
            if resposta_usuario and resposta_usuario != 'nan' and resposta_usuario != '':
                total += 1
                if resposta_usuario == resposta_correta.strip().lower():
                    acertos += 1
        
        if total > 0:
            percentual_acerto = (acertos / total) * 100
            percentual_erro = 100 - percentual_acerto
        else:
            percentual_acerto = 0
            percentual_erro = 0
            
        estatisticas.append({
            'Pergunta': pergunta_gabarito,
            'Acertos (%)': round(percentual_acerto, 2),
            'Erros (%)': round(percentual_erro, 2),
            'Total_Respostas': total,
            'Coluna_Utilizada': coluna_encontrada
        })
    
    return estatisticas

# =========================
# Cache de dados
# =========================

@st.cache_data
def load_data():
    admitidos = pd.read_excel("data/Admitidos.xlsx")
    integracao = {m: pd.read_excel("data/Integracao Armazem.xlsx", sheet_name=m) for m in ["M1", "M2", "M3", "M4", "M5"]}
    return admitidos, integracao

# =========================
# Carregamento dos dados
# =========================
with st.spinner('Carregando dados...'):
    admitidos, integracao = load_data()

# =========================
# Processamento inicial
# =========================
admitidos = admitidos[admitidos['Cargo'] == "Ajudante Armazém"]
admitidos = admitidos[~admitidos['Operação'].isin(["VIDROS PR", "PONTA GROSSA"])]
admitidos['Data'] = pd.to_datetime(admitidos['Data'], errors='coerce')

# Regras específicas por unidade
data_corte_petropolis = datetime(2025, 10, 1)
data_corte_litoral = datetime(2024, 11, 1)

admitidos = admitidos[
    ~(
        (admitidos['Operação'] == "CD PETRÓPOLIS") & (admitidos['Data'] < data_corte_petropolis)
    )
]
admitidos = admitidos[
    ~(
        (admitidos['Operação'] == "CD LITORAL") & (admitidos['Data'] < data_corte_litoral)
    )
]

hoje_corte = datetime(2026, 2, 28)
if datetime.now() <= hoje_corte:
    admitidos = admitidos[admitidos['Turno'].str.upper() == "NOITE"]

# =========================
# Sidebar - Filtros
# =========================
with st.sidebar:
    st.markdown("## 🎛️ Filtros")
    
    # Lista suspensa para Operação
    operacoes = sorted(admitidos['Operação'].unique())
    filtro_operacao = st.selectbox(
        "🏭 Operação", 
        ["Todas"] + operacoes,
        help="Selecione uma operação específica"
    )
    
    # Datas separadas (início e fim)
    min_date = admitidos['Data'].min().date()
    max_date = admitidos['Data'].max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        data_inicio = st.date_input(
            "📅 Data inicial", 
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        data_fim = st.date_input(
            "📅 Data final", 
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Lista suspensa para Colaborador
    colaboradores = sorted(admitidos['Colaborador'].unique())
    filtro_colaborador = st.selectbox(
        "👤 Colaborador", 
        ["Todos"] + colaboradores,
        help="Selecione um colaborador específico"
    )

# =========================
# Aplicando filtros
# =========================
admitidos_filtrado = admitidos.copy()

# Filtro de operação
if filtro_operacao != "Todas":
    admitidos_filtrado = admitidos_filtrado[admitidos_filtrado['Operação'] == filtro_operacao]

# Filtro de data
admitidos_filtrado = admitidos_filtrado[
    (admitidos_filtrado['Data'] >= pd.to_datetime(data_inicio)) & 
    (admitidos_filtrado['Data'] <= pd.to_datetime(data_fim))
]

# Filtro de colaborador
if filtro_colaborador != "Todos":
    admitidos_filtrado = admitidos_filtrado[
        admitidos_filtrado['Colaborador'] == filtro_colaborador
    ]

# =========================
# Fuzzy Match
# =========================
results_list = []

for modulo, df_mod in integracao.items():
    for _, row in admitidos_filtrado.iterrows():
        matched = fuzzy_match(row, df_mod)
        status = "Realizado" if matched is not None else "Pendente"
        results_list.append({
            "Operação": row['Operação'],
            "Colaborador": row['Colaborador'],
            "Data": row['Data'],
            "Módulo": modulo,
            "Status": status
        })

resultado_modulos = pd.DataFrame(results_list)

# =========================
# Título e gráfico de aderência geral
# =========================
st.title("📦 Dashboard de Integração Armazém")

# Gráfico de aderência geral por operação
if len(resultado_modulos) > 0:
    aderencia_geral = resultado_modulos.groupby('Operação')['Status'].apply(
        lambda x: (x == "Realizado").mean() * 100
    ).reset_index()
    aderencia_geral.columns = ['Operação', 'Aderência (%)']
    
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
    fig_aderencia.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_aderencia.update_layout(
        xaxis_title="Operação",
        yaxis_title="Aderência (%)",
        yaxis_range=[0, 100],
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig_aderencia, use_container_width=True, key="aderencia_geral")
    
    # Card de quantidade de realizados
    total_realizados = len(resultado_modulos[resultado_modulos['Status'] == 'Realizado'].drop_duplicates(subset=['Colaborador', 'Módulo']))
    st.metric("✅ Qtde Realizados", total_realizados)
else:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")

st.markdown("---")

# =========================
# Tabela detalhada
# =========================
st.subheader("📄 Detalhamento por Colaborador")

if len(resultado_modulos) > 0:
    # Lista suspensa para filtro de módulo (com todos selecionados por padrão)
    modulos_disponiveis = ["M1", "M2", "M3", "M4", "M5"]
    filtro_modulo_detalhe = st.selectbox(
        "Filtrar por módulo",
        options=["Todos"] + modulos_disponiveis,
        index=0,
        key="filtro_modulo"
    )
    
    if filtro_modulo_detalhe == "Todos":
        tabela_filtrada = resultado_modulos.copy()
    else:
        tabela_filtrada = resultado_modulos[resultado_modulos['Módulo'] == filtro_modulo_detalhe]
    
    # Formatar data para formato reduzido
    if 'Data' in tabela_filtrada.columns:
        tabela_filtrada['Data'] = pd.to_datetime(tabela_filtrada['Data']).dt.strftime('%d/%m/%Y')
    
    # Tabela com cores
    def color_status(val):
        if val == 'Realizado':
            return 'background-color: #28a745; color: white'
        else:
            return 'background-color: #dc3545; color: white'
    
    styled_table = tabela_filtrada.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_table, use_container_width=True, height=400)
else:
    st.info("Nenhum dado para exibir na tabela detalhada.")

st.markdown("---")

# =========================
# Análise de Respostas Check de Retenção
# =========================
st.header("📝 Análise Respostas Check de Retenção")

# Gabaritos com as perguntas e respostas corretas
gabaritos = {
    "M1": {
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

# Criar abas para cada módulo
modulos_lista = list(integracao.keys())
tabs = st.tabs([f"📘 {modulo}" for modulo in modulos_lista])

# Iterar corretamente sobre as abas
for idx, modulo in enumerate(modulos_lista):
    with tabs[idx]:
        st.subheader(f"Análise de Respostas - {modulo}")
        
        df_mod = integracao[modulo]
        
        # Verificar se o módulo existe no gabarito
        if modulo not in gabaritos:
            st.warning(f"Gabarito não encontrado para o módulo {modulo}")
            continue
            
        respostas_validas = gabaritos[modulo]['respostas']
        perguntas_gabarito = gabaritos[modulo]['perguntas']
        
        # Mostrar as colunas disponíveis para debug (opcional, pode remover depois)
        with st.expander("🔍 Informações de depuração", expanded=False):
            st.write(f"Colunas disponíveis no módulo {modulo}:")
            colunas_df = [col for col in df_mod.columns if col not in ['Colaborador', 'CPF']]
            st.write(colunas_df)
            st.write(f"Número de perguntas no gabarito: {len(perguntas_gabarito)}")
            st.write(f"Número de colunas no DataFrame: {len(colunas_df)}")
        
        # Calcular estatísticas por pergunta
        estatisticas = calcular_estatisticas_respostas(df_mod, perguntas_gabarito, respostas_validas)
        
        # Criar gráficos para cada pergunta
        for i, pergunta_data in enumerate(estatisticas, 1):
            with st.expander(f"📌 {i}. {pergunta_data['Pergunta']}", expanded=True):
                # Criar dataframe para o gráfico
                df_pergunta = pd.DataFrame({
                    'Status': ['Acertos', 'Erros'],
                    'Percentual': [pergunta_data['Acertos (%)'], pergunta_data['Erros (%)']]
                })
                
                # Gráfico de barras verticais
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
                    height=400,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Percentual (%)"
                )
                
                # Usar key única para cada gráfico
                chart_key = f"chart_{modulo}_{i}_{datetime.now().timestamp()}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
                # Mostrar métricas adicionais
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Acertos", f"{pergunta_data['Acertos (%)']:.1f}%")
                with col2:
                    st.metric("❌ Erros", f"{pergunta_data['Erros (%)']:.1f}%")
                
                st.caption(f"📊 Total de respostas analisadas: {pergunta_data['Total_Respostas']}")
                if 'Coluna_Utilizada' in pergunta_data:
                    st.caption(f"🔍 Coluna correspondente: {pergunta_data['Coluna_Utilizada']}")
                st.markdown("---")

st.markdown("### 📌 Legenda")
col_leg1, col_leg2, col_leg3 = st.columns(3)
with col_leg1:
    st.markdown("🟩 **Acertos** - Respostas corretas")
with col_leg2:
    st.markdown("🟥 **Erros** - Respostas incorretas")
with col_leg3:
    st.markdown("📊 **Total** - Quantidade de respostas analisadas")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Configuração da página
# =========================
st.set_page_config(
    page_title="Dashboard Integração Armazém", 
    layout="wide",
    page_icon="📦",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhor visual
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
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

def criar_grafico_gauge(valor, titulo):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = valor,
        title = {'text': titulo},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 60], 'color': "#ff6b6b"},
                {'range': [60, 85], 'color': "#ffd93d"},
                {'range': [85, 100], 'color': "#6bcf7f"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

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
    st.image("https://via.placeholder.com/300x100?text=Logo", use_column_width=True)
    st.markdown("## 🎛️ Filtros")
    
    operacoes = sorted(admitidos['Operação'].unique())
    filtro_operacao = st.multiselect(
        "🏭 Operação", 
        operacoes, 
        operacoes,
        help="Selecione uma ou mais operações"
    )
    
    min_date, max_date = admitidos['Data'].min(), admitidos['Data'].max()
    filtro_data = st.date_input(
        "📅 Data de admissão", 
        value=(min_date, max_date),
        help="Selecione o período desejado"
    )
    
    filtro_colaborador = st.text_input(
        "👤 Colaborador específico",
        placeholder="Digite o nome...",
        help="Busque por um colaborador específico"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Opções de visualização")
    mostrar_detalhado = st.checkbox("Mostrar tabela detalhada", value=True)
    mostrar_graficos = st.checkbox("Mostrar gráficos avançados", value=True)

# =========================
# Aplicando filtros
# =========================
admitidos_filtrado = admitidos.copy()
admitidos_filtrado = admitidos_filtrado[admitidos_filtrado['Operação'].isin(filtro_operacao)]
admitidos_filtrado = admitidos_filtrado[
    (admitidos_filtrado['Data'] >= pd.to_datetime(filtro_data[0])) & 
    (admitidos_filtrado['Data'] <= pd.to_datetime(filtro_data[1]))
]

if filtro_colaborador:
    admitidos_filtrado = admitidos_filtrado[
        admitidos_filtrado['Colaborador'].str.contains(filtro_colaborador, case=False, na=False)
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
# Métricas principais
# =========================
st.title("📦 Dashboard de Integração Armazém")
st.markdown(f"📅 **Última atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_colaboradores = len(admitidos_filtrado)
    st.metric("👥 Total de Colaboradores", total_colaboradores, delta=None)

with col2:
    total_operacoes = len(admitidos_filtrado['Operação'].unique())
    st.metric("🏭 Operações", total_operacoes)

with col3:
    total_modulos = len(integracao)
    st.metric("📚 Módulos", total_modulos)

with col4:
    taxa_geral = (resultado_modulos['Status'] == "Realizado").mean() * 100
    st.metric("✅ Taxa Geral", f"{taxa_geral:.1f}%", 
              delta=f"{taxa_geral - 50:.1f}%" if taxa_geral > 50 else None)

st.markdown("---")

# =========================
# Gráficos de aderência
# =========================
if mostrar_graficos:
    st.subheader("📊 Análise de Aderência")
    
    # Preparação dos dados
    aderencia = resultado_modulos.groupby(['Operação', 'Módulo'])['Status'].apply(
        lambda x: (x == "Realizado").mean()
    ).reset_index()
    aderencia.rename(columns={"Status": "Aderência"}, inplace=True)
    aderencia['Aderência (%)'] = aderencia['Aderência'] * 100
    
    # Gráfico de barras interativo
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_bar = px.bar(
            aderencia, 
            x='Operação', 
            y='Aderência (%)', 
            color='Módulo',
            title='Aderência por Operação e Módulo',
            text='Aderência (%)',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(
            xaxis_title="Operação",
            yaxis_title="Aderência (%)",
            yaxis_range=[0, 100],
            hovermode='x unified'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Heatmap simplificado
        pivot_aderencia = aderencia.pivot(index='Módulo', columns='Operação', values='Aderência (%)')
        fig_heatmap = px.imshow(
            pivot_aderencia,
            text_auto='.1f',
            aspect="auto",
            color_continuous_scale='RdYlGn',
            title='Heatmap de Aderência'
        )
        fig_heatmap.update_layout(height=300)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Gráficos de pizza por módulo
    st.subheader("🥧 Distribuição por Módulo")
    cols_pizza = st.columns(3)
    
    for idx, modulo in enumerate(["M1", "M2", "M3", "M4", "M5"]):
        if idx < len(cols_pizza):
            with cols_pizza[idx % 3]:
                modulo_data = resultado_modulos[resultado_modulos['Módulo'] == modulo]
                status_counts = modulo_data['Status'].value_counts()
                
                fig_pie = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title=f'{modulo}',
                    color=status_counts.index,
                    color_discrete_map={'Realizado': '#28a745', 'Pendente': '#dc3545'},
                    hole=0.3
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# Tabela de aderência estilizada
# =========================
st.subheader("📈 Resumo de Aderência por Operação")

def color_adherence(val):
    if val >= 95:
        return 'background-color: #28a745; color: white'
    elif val >= 85:
        return 'background-color: #ffc107; color: black'
    else:
        return 'background-color: #dc3545; color: white'

aderencia_style = aderencia[['Operação', 'Módulo', 'Aderência (%)']].copy()
aderencia_style['Aderência (%)'] = aderencia_style['Aderência (%)'].round(1)

pivot_table = aderencia_style.pivot(index='Operação', columns='Módulo', values='Aderência (%)')
styled_pivot = pivot_table.style.background_gradient(cmap='RdYlGn', axis=None, vmin=0, vmax=100)
styled_pivot = styled_pivot.format("{:.1f}%")
st.dataframe(styled_pivot, use_container_width=True, height=400)

# =========================
# Tabela detalhada
# =========================
if mostrar_detalhado:
    st.markdown("---")
    st.subheader("📄 Detalhamento por Colaborador")
    
    # Adicionar filtro de módulo na tabela detalhada
    filtro_modulo_detalhe = st.multiselect(
        "Filtrar por módulo",
        options=["M1", "M2", "M3", "M4", "M5"],
        default=["M1", "M2", "M3", "M4", "M5"]
    )
    
    tabela_filtrada = resultado_modulos[resultado_modulos['Módulo'].isin(filtro_modulo_detalhe)]
    
    # Tabela com cores
    def color_status(val):
        if val == 'Realizado':
            return 'background-color: #28a745; color: white'
        else:
            return 'background-color: #dc3545; color: white'
    
    styled_table = tabela_filtrada.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_table, use_container_width=True, height=500)
    
    # Botão de exportação
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        csv = resultado_modulos.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name=f"status_integracao_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_btn2:
        # Simular exportação Excel (precisa de openpyxl)
        st.download_button(
            label="📊 Baixar Excel",
            data=csv,
            file_name=f"status_integracao_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# =========================
# Parte 2 - Correção das respostas
# =========================
st.markdown("---")
st.header("📝 Análise de Acertos e Erros por Módulo")

# Gabaritos
gabaritos = {
    "M1": {
        "pontuacao": 90,
        "respostas": [
            "Cinto paraquedista anti queda",
            "Todo barril deve ser movimentado por duas pessoas utilizando os EPIs corretos",
            "Durante as atividades do Picking, deve-se manter a segregação homem x máquina",
            "Validar a condição do palete e, em caso de más condições, o palete de madeira deve ser segregado e substituído",
            "Devemos realizar a movimentação de garrafas de forma individual e com as duas mãos",
            "Deve-se garantir que o local está limpo e sem a presença de nenhum obstáculo que atrapalhe a movimentação dos produtos",
            "Os produtos com data curta devem estar disponíveis antes das datas longas, desde que estejam dentro do prazo comercial",
            "Podemos realizar a reembalagem de produtos com diferentes datas de validade",
            "A chave do caminhão nunca pode estar na ignição",
        ]
    },
    "M2": {
        "pontuacao": 40,
        "respostas": [
            "Curva A: \"Alto giro\"; Curva B: \"Médio giro\"; Curva C: \"Baixo giro\"",
            "Todas as alternativas estão corretas",
            "O Picking é a área que é projetada para realizarmos a montagem dos paletes durante a separação",
            "O layout ideal do Picking é organizar as posições baseadas por embalagem"
        ]
    },
    "M3": {
        "pontuacao": 40,
        "respostas": [
            "Os produtos mistos devem ser colocados nas laterais do palete para facilitar a conferência",
            "Palete de madeira, lastro e elemento divisório (chapatex e folha separadora)",
            "Os produtos diversos precisam ser sempre colocados na parte mais externa do lastro para que estejam visíveis durante a conferência",
            "Após a separação do palete, o mesmo é conferido pelo conferente e, em seguida, liberado para carregamento"
        ]
    },
    "M4": {
        "pontuacao": 40,
        "respostas": [
            "Caso tenha problemas com meu usuário devo utilizar a função \"Esqueci a senha\" ou procurar suporte do conferente",
            "Não é mostrada a quantidade correta do produto na tela do WMS",
            "Devemos ter atenção na descrição por completo",
            "Associar palete, visualizar produto, realizar separação, conferir e finalizar palete"
        ]
    },
    "M5": {
        "pontuacao": 40,
        "respostas": [
            "Há uma memória de cálculo para classificar a complexidade dos paletes",
            "Quanto mais montar, maior a remuneração variável",
            "Ponto laboral, PPS (Ponto por segundo) e EFM (Eficiência de Montagem)",
            "Separar a quantidade correta, do produto correto e em boas condições"
        ]
    },
}

# Criar abas para cada módulo
tabs = st.tabs([f"📘 {modulo}" for modulo in integracao.keys()])

for tab, (modulo, df_mod) in zip(tabs, integracao.items()):
    with tab:
        respostas_validas = gabaritos[modulo]['respostas']
        mod_copy = df_mod.copy()
        
        perguntas = [c for c in mod_copy.columns if c not in ['Colaborador', 'CPF']]
        
        acertos = []
        erros = []
        
        for _, row in mod_copy.iterrows():
            ac = 0
            er = 0
            for p, correta in zip(perguntas, respostas_validas):
                if str(row[p]).strip().lower() == correta.strip().lower():
                    ac += 1
                else:
                    er += 1
            acertos.append(ac)
            erros.append(er)
        
        mod_copy['Acertos'] = acertos
        mod_copy['Erros'] = erros
        mod_copy['% Acerto'] = (mod_copy['Acertos'] / len(respostas_validas) * 100).round(1)
        
        # Métricas do módulo
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Colaboradores", len(mod_copy))
        with col2:
            media_acertos = mod_copy['% Acerto'].mean()
            st.metric("Média de Acertos", f"{media_acertos:.1f}%")
        with col3:
            melhor_acerto = mod_copy['% Acerto'].max()
            st.metric("Melhor Performance", f"{melhor_acerto:.1f}%")
        with col4:
            pior_acerto = mod_copy['% Acerto'].min()
            st.metric("Pior Performance", f"{pior_acerto:.1f}%")
        
        # Gráfico de distribuição
        fig_hist = px.histogram(
            mod_copy, 
            x='% Acerto',
            nbins=20,
            title=f'Distribuição de Acertos - {modulo}',
            labels={'% Acerto': 'Percentual de Acerto (%)', 'count': 'Número de Colaboradores'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.add_vline(x=media_acertos, line_dash="dash", line_color="red", 
                          annotation_text=f"Média: {media_acertos:.1f}%")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Tabela com ordenação
        ordenar_por = st.selectbox(
            f"Ordenar por - {modulo}",
            ['% Acerto', 'Acertos', 'Erros'],
            key=f"sort_{modulo}"
        )
        
        tabela_ordenada = mod_copy[['Colaborador', 'CPF', 'Acertos', 'Erros', '% Acerto']].sort_values(
            by=ordenar_por, ascending=False
        )
        
        # Estilizar a tabela
        def color_percent(val):
            if val >= 80:
                return 'background-color: #28a745; color: white'
            elif val >= 60:
                return 'background-color: #ffc107; color: black'
            else:
                return 'background-color: #dc3545; color: white'
        
        styled_table = tabela_ordenada.style.applymap(color_percent, subset=['% Acerto'])
        st.dataframe(styled_table, use_container_width=True, height=400)

st.markdown("---")
st.markdown("### 📌 Legenda")
col_leg1, col_leg2, col_leg3 = st.columns(3)
with col_leg1:
    st.markdown("🟩 **≥95%** - Excelente")
    st.markdown("🟨 **85-95%** - Bom")
with col_leg2:
    st.markdown("🟧 **70-85%** - Regular")
    st.markdown("🟥 **<70%** - Crítico")
with col_leg3:
    st.markdown("✅ **Realizado** - Módulo completo")
    st.markdown("❌ **Pendente** - Módulo não iniciado")

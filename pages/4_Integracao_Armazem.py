import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher

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


# =========================
# Config Streamlit
# =========================

st.set_page_config(page_title="Integração Armazém", layout="wide")
st.title("📦 Integração Armazém - Acompanhamento de Módulos")

# =========================
# Importação das bases
# =========================

@st.cache_data
def load_data():
    admitidos = pd.read_excel("data/Admitidos.xlsx")
    integracao = {m: pd.read_excel("data/Integracao Armazem.xlsx", sheet_name=m) for m in ["M1", "M2", "M3", "M4", "M5"]}
    return admitidos, integracao

admitidos, integracao = load_data()

# =========================
# Filtros iniciais da base Admitidos
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

# Regras do turno
hoje_corte = datetime(2026, 2, 28)
if datetime.now() <= hoje_corte:
    admitidos = admitidos[admitidos['Turno'].str.upper() == "NOITE"]

# =========================
# Filtros do usuário
# =========================

st.sidebar.header("Filtros")
operacoes = sorted(admitidos['Operação'].unique())
filtro_operacao = st.sidebar.multiselect("Operação", operacoes, operacoes)

min_date, max_date = admitidos['Data'].min(), admitidos['Data'].max()
filtro_data = st.sidebar.date_input("Data de admissão (intervalo)", value=(min_date, max_date))

filtro_colaborador = st.sidebar.text_input("Colaborador específico (opcional)")

# Aplicando filtros do usuário
admitidos_filtrado = admitidos.copy()
admitidos_filtrado = admitidos_filtrado[admitidos_filtrado['Operação'].isin(filtro_operacao)]
admitidos_filtrado = admitidos_filtrado[(admitidos_filtrado['Data'] >= pd.to_datetime(filtro_data[0])) & (admitidos_filtrado['Data'] <= pd.to_datetime(filtro_data[1]))]

if filtro_colaborador:
    admitidos_filtrado = admitidos_filtrado[admitidos_filtrado['Colaborador'].str.contains(filtro_colaborador, case=False, na=False)]

# =========================
# Fuzzy Match com as abas de integração
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
# Gráfico de aderência por operação
# =========================

st.subheader("📊 Aderência por Operação")

aderencia = resultado_modulos.groupby(['Operação', 'Módulo'])['Status'].apply(lambda x: (x == "Realizado").mean()).reset_index()
aderencia.rename(columns={"Status": "Aderência"}, inplace=True)

# Aplicar coloração

def cor(valor):
    if valor >= 0.95:
        return "🟩"
    elif valor >= 0.85:
        return "🟨"
    else:
        return "🟥"

aderencia['Cor'] = aderencia['Aderência'].apply(cor)

st.dataframe(aderencia)

# =========================
# Tabela detalhada + exportação
# =========================

st.subheader("📄 Detalhamento por Colaborador")
st.dataframe(resultado_modulos)

excel = st.download_button(
    label="📥 Baixar Excel",
    data=resultado_modulos.to_csv(index=False).encode('utf-8'),
    file_name="status_integracao.csv",
    mime="text/csv"
)

# =========================
# Parte 2 - Correção das respostas
# =========================

st.header("📝 Acertos e Erros por Módulo")

# Gabaritos fixos

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

# Calcular acertos e erros por módulo

for modulo, df_mod in integracao.items():
    st.subheader(f"📘 {modulo}")

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
    mod_copy['% Acerto'] = mod_copy['Acertos'] / len(respostas_validas)

    st.dataframe(mod_copy[['Colaborador', 'CPF', 'Acertos', 'Erros', '% Acerto']])

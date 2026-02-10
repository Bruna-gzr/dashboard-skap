import streamlit as st

st.title("🆕 Acompanhamento de Novos")

st.sidebar.header("Filtros")
st.sidebar.multiselect("Unidade", [])
st.sidebar.multiselect("Gestor", [])
st.sidebar.multiselect("Status", ["0-15 dias", "16-30 dias", "31-60 dias", "61-90 dias", "Acima de 90 dias"])

st.info("Em construção — aqui vamos montar a base e os checkpoints.")

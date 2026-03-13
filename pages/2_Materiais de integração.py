def criar_card_unidade(nome_unidade, dados):
    with st.container(border=True):

        # ===== LOGO + TITULO CENTRALIZADOS NO MESMO EIXO =====
        esp1, centro, esp2 = st.columns([1, 2, 1])

        with centro:
            col_a, col_b, col_c = st.columns([1, 2, 1])

            with col_b:
                try:
                    st.image(dados["logo"], width=120)
                except Exception:
                    st.markdown(
                        """
                        <div style='
                            background: white;
                            border-radius: 50%;
                            width: 120px;
                            height: 120px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin: 0 auto;
                        '>
                            <span style='font-size: 60px;'>🏢</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"<div class='titulo-unidade'>{nome_unidade}</div>",
                    unsafe_allow_html=True
                )

        # Duas colunas para os setores
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"<div class='titulo-coluna'>{dados['coluna1']['titulo']}</div>",
                unsafe_allow_html=True
            )
            for setor in dados["coluna1"]["setores"]:
                icone = ICONES.get(setor, "🔗")
                if st.button(
                    f"{icone} {setor}",
                    key=f"{nome_unidade}_dist_{setor}",
                    use_container_width=True
                ):
                    st.info(f"Link para {setor}")

        with col2:
            st.markdown(
                f"<div class='titulo-coluna'>{dados['coluna2']['titulo']}</div>",
                unsafe_allow_html=True
            )
            for setor in dados["coluna2"]["setores"]:
                icone = ICONES.get(setor, "🔗")
                if st.button(
                    f"{icone} {setor}",
                    key=f"{nome_unidade}_arm_{setor}",
                    use_container_width=True
                ):
                    st.info(f"Link para {setor}")

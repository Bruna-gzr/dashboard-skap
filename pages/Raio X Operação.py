        # ---------------------------
        # ✅ FIX: Streamlit pode quebrar com Styler + colunas duplicadas.
        # Vamos achatar e renomear, mantendo "PTS" visualmente repetido
        # usando sufixos invisíveis (zero-width) para ficar ÚNICO.
        # ---------------------------
        out_flat = out.copy()

        # 1) achata MultiIndex
        out_flat.columns = [
            (c[0] if (isinstance(c, tuple) and c[1] == "") else f"{c[0]}_{c[1]}")
            for c in out_flat.columns
        ]

        # 2) renomeia conforme pedido:
        #    *_Resultado -> sem sufixo
        #    *_PTS -> "PTS" (com sufixo invisível único)
        ZERO = "\u200b"  # zero-width space (invisível)

        def col_indicador_base(col: str) -> str:
            # pega o "nome do indicador" antes do sufixo
            if col.endswith("_PTS"):
                return col[:-4]
            if col.endswith("_Resultado"):
                return col[:-10]
            return col

        novos = []
        for col in out_flat.columns:
            col = str(col)

            if col.endswith("_Resultado"):
                novos.append(col.replace("_Resultado", ""))  # JL_Resultado -> JL
                continue

            if col.endswith("_PTS"):
                ind = col_indicador_base(col)               # JL_PTS -> JL
                # cria um PTS "visualmente igual", mas único por indicador:
                # PTS + (zero-width * hash)
                # (pode ser por comprimento do indicador ou por índice; aqui vai por hash estável)
                suffix_len = (abs(hash(ind)) % 6) + 1       # 1..6 (pequeno)
                novos.append("PTS" + (ZERO * suffix_len))
                continue

            # Recargas_Resultado -> Recargas já cai no caso "_Resultado"
            # Média RV_Resultado -> Média RV já cai no caso "_Resultado"
            novos.append(col)

        out_flat.columns = novos

        # styler central
        sty = out_flat.style.set_properties(**{"text-align": "center"}).set_table_styles(
            [{"selector": "th", "props": [("text-align", "center")]}]
        )

        # colunas PTS agora são as que começam com "PTS"
        pts_cols = [c for c in out_flat.columns if str(c).startswith("PTS")]
        if pts_cols:
            sty = sty.applymap(color_pts_zero, subset=pts_cols)

        # risco
        if "RISCO DE TO?" in out_flat.columns:
            sty = sty.applymap(color_risco, subset=["RISCO DE TO?"])

        # RV (agora é "Média RV")
        if "Média RV" in out_flat.columns:
            sty = sty.applymap(color_rv_cell, subset=["Média RV"])

        st.dataframe(sty, use_container_width=True, height=520)

        # Export (usa o out_flat direto)
        excel = preparar_excel_para_download(out_flat, sheet_name="RAIO_X")
        st.download_button(
            "⬇️ Baixar Excel (Tabela RAIO X)",
            data=excel,
            file_name="raio_x_operacao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

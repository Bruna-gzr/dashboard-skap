st.markdown("""
<style>

.stApp{
    background:#050816;
}

.page-title{
    text-align:center;
    color:white;
    font-size:32px;
    margin-bottom:25px;
}

/* colunas com mesma altura */
div[data-testid="column"]{
    display:flex;
    align-items:stretch;
}

div[data-testid="column"] > div{
    width:100%;
    display:flex;
    flex-direction:column;
}

/* card */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:linear-gradient(135deg,#2D2D2D 0%,#404040 100%);
    border-radius:20px;
    padding:25px 20px;
    margin:8px 0;
    border:1px solid #555;
    box-shadow:0 10px 30px rgba(0,0,0,0.3);
    min-height:660px;
    height:660px;
    width:100%;
    box-sizing:border-box;
    display:flex;
    flex-direction:column;
    overflow:hidden;
}

/* cabeçalho */
.unidade-header{
    height:235px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:flex-start;
    flex-shrink:0;
}

/* imagem */
div[data-testid="stImage"]{
    display:flex;
    justify-content:center;
    align-items:center;
    min-height:160px;
}

div[data-testid="stImage"] img{
    display:block;
    margin:0 auto;
    object-fit:contain;
}

/* título */
.unidade-titulo{
    min-height:42px;
    display:flex;
    align-items:center;
    justify-content:center;
    text-align:center;
    color:white;
    font-size:26px;
    font-weight:700;
    margin:0;
    padding:0 8px;
    text-transform:uppercase !important;
}

/* área de conteúdo */
.conteudo-card{
    flex:1;
    display:flex;
    flex-direction:column;
    justify-content:flex-start;
}

/* quando o conteúdo passar, rola só dentro */
.scroll-card{
    flex:1;
    overflow-y:auto;
    overflow-x:hidden;
    padding-right:4px;
}

/* barra de rolagem */
.scroll-card::-webkit-scrollbar{
    width:6px;
}
.scroll-card::-webkit-scrollbar-thumb{
    background:#666;
    border-radius:10px;
}
.scroll-card::-webkit-scrollbar-track{
    background:transparent;
}

/* títulos de coluna */
.titulo-coluna{
    color:#CCCCCC;
    font-weight:bold;
    margin-bottom:15px;
    font-size:16px;
    text-align:center;
    width:100%;
}

/* botões */
.stButton button{
    width:100%;
    background:#3A3A3A;
    color:#FFFFFF;
    border:1px solid #555555;
    border-radius:8px;
    padding:8px 12px;
    font-size:14px;
    font-weight:500;
    text-align:left;
    margin:3px 0;
    transition:0.2s ease;
}

.stButton button:hover{
    background:#4A4A4A;
    border:1px solid #777777;
    transform:translateY(-1px);
}

/* fallback */
.logo-fallback{
    background:white;
    border-radius:50%;
    width:120px !important;
    height:120px !important;
    display:flex;
    align-items:center;
    justify-content:center;
    margin:0 auto 15px auto;
}

.logo-fallback-grande{
    background:white;
    border-radius:50%;
    width:160px !important;
    height:160px !important;
    display:flex;
    align-items:center;
    justify-content:center;
    margin:0 auto 15px auto;
}

.logo-fallback span,
.logo-fallback-grande span{
    font-size:70px;
}

</style>
""", unsafe_allow_html=True)

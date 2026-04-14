import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# 1. Configuração da Página
st.set_page_config(page_title="Defesa Civil - IFMT Campo Verde", layout="wide", page_icon="🛡️")

st.title(" Monitoramento e Previsão Climática")
st.markdown("---")

@st.cache_data
def carregar_e_limpar_dados():
    if not os.path.exists('dados_estacao_limpos.csv'):
        return None
        
    # Carrega pulando a linha em inglês
    df = pd.read_csv('dados_estacao_limpos.csv', skiprows=[1], low_memory=False)
    
    # LIMPEZA DE COLUNAS: Remove quebras de linha e espaços duplos invisíveis
    df.columns = [" ".join(col.replace('\n', ' ').split()) for col in df.columns]
    
    # Limpa linhas que repetem o cabeçalho
    df = df[~df['Data'].str.contains('Date|Data', na=False)]
    
    # CONVERSÃO DE TEMPO: Trata o formato específico da estação (YYYY-MM-DD)
    df['timestamp'] = pd.to_datetime(
        df['Data'].astype(str).str.split(' ').str[0] + ' ' + df['Hora'].astype(str), 
        format='%Y-%m-%d %H:%M:%S', 
        errors='coerce'
    )
    df = df.dropna(subset=['timestamp'])
    
    # Conversão Numérica (Trata vírgulas e pontos)
    cols_num = ['Temperatura Externa', 'Umidade do Ar Externa', 'Pluviômetro (Chuva ) mm', 'Radiação Solar (Wm2)']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    return df.sort_values('timestamp')

try:
    df = carregar_e_limpar_dados()
    
    if df is None:
        st.error(" Arquivo 'dados_estacao_limpos.csv' não encontrado!")
        st.stop()

    # 2. Barra Lateral - Filtros e IA
    st.sidebar.header("📅 Configurações")
    
    data_min = df['timestamp'].min().date()
    data_max = df['timestamp'].max().date()
    
    selecao = st.sidebar.date_input(
        "Período de Análise", 
        value=[data_min, data_max], 
        min_value=data_min, 
        max_value=data_max
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Inteligência Artificial")
    btn_previsao = st.sidebar.button("Prever Próxima Hora")

    # 3. Processamento de Filtros (Proteção contra NaTType)
    if isinstance(selecao, (list, tuple)) and len(selecao) == 2:
        inicio, fim = selecao
        
        mask = (df['timestamp'].dt.date >= inicio) & (df['timestamp'].dt.date <= fim)
        df_f = df.loc[mask]

        if not df_f.empty:
            st.subheader(f" Diagnóstico: {inicio.strftime('%d/%m/%Y')} até {fim.strftime('%d/%m/%Y')}")
            
            c1, c2, c3, c4 = st.columns(4)
            
            # Buscando colunas limpas
            t_max = df_f['Temperatura Externa'].max()
            u_min = df_f['Umidade do Ar Externa'].min()
            chuva = df_f['Pluviômetro (Chuva ) mm'].sum()
            rad = df_f['Radiação Solar (Wm2)'].max()

            c1.metric("Máxima Térmica", f"{t_max:.1f} °C" if pd.notnull(t_max) else "N/A")
            c2.metric("Umidade Mínima", f"{u_min:.1f}%" if pd.notnull(u_min) else "N/A")
            c3.metric("Chuva Acumulada", f"{chuva:.1f} mm" if pd.notnull(chuva) else "0.0 mm")
            c4.metric("Radiação Solar", f"{rad:.0f} W/m²" if pd.notnull(rad) else "N/A")

            # 4. Visualização
            st.markdown("---")
            aba1, aba2 = st.tabs(["🌡️ Termo-Higrométrico", "🌧️ Pluviometria"])

            with aba1:
                fig_temp = px.line(df_f, x='timestamp', y=['Temperatura Externa', 'Umidade do Ar Externa'],
                                  title="Evolução: Temperatura vs Umidade",
                                  color_discrete_sequence=['#e74c3c', '#3498db'])
                st.plotly_chart(fig_temp, use_container_width=True)

            with aba2:
                fig_chuva = px.bar(df_f, x='timestamp', y='Pluviômetro (Chuva ) mm', 
                                   title="Precipitação (mm)",
                                   color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig_chuva, use_container_width=True)

            # Alerta Crítico
            if pd.notnull(u_min) and u_min < 20:
                st.warning(" **ALERTA:** Umidade muito baixa! Risco elevado de incêndios florestais.")
        else:
            st.warning("Nenhum registro encontrado para este período.")
    else:
        st.info(" Selecione o intervalo de datas no menu lateral.")

    # 5. Inteligência Artificial
    if btn_previsao:
        if os.path.exists('modelo_clima_ifmt.pkl'):
            modelo = joblib.load('modelo_clima_ifmt.pkl')
            # Pega os dados mais recentes para a predição
            atual = df.iloc[-1]
            anterior = df.iloc[-2]
            
            X_input = pd.DataFrame([[
                atual['Temperatura Externa'], atual['Umidade do Ar Externa'],
                anterior['Temperatura Externa'], anterior['Umidade do Ar Externa'],
                atual['Radiação Solar (Wm2)']
            ]], columns=['Temperatura Externa', 'Umidade do Ar Externa', 'temp_anterior', 'umid_anterior', 'rad_anterior'])
            
            pred = modelo.predict(X_input)[0]
            st.sidebar.success(f" Estimativa Próxima Hora: **{pred:.1f} °C**")
        else:
            st.sidebar.error("Modelo IA não encontrado. Treine o modelo primeiro.")

except Exception as e:
    st.error(f"Erro inesperado: {e}")
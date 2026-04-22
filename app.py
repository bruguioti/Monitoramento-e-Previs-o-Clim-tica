import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from datetime import datetime

# 1. Configuração da Página
st.set_page_config(page_title="IFMT Campo Verde", layout="wide", page_icon="🌱")

st.title(" Monitoramento Agrometeorológico e Gestão de Riscos")
st.markdown("---")

@st.cache_data
def carregar_dados():
    if not os.path.exists('dados_estacao_limpos.csv'): return None
    df = pd.read_csv('dados_estacao_limpos.csv', skiprows=[1], low_memory=False)
    df.columns = [" ".join(col.replace('\n', ' ').split()) for col in df.columns]
    df = df[~df['Data'].str.contains('Date|Data', na=False)]
    df['timestamp'] = pd.to_datetime(df['Data'].astype(str).str.split(' ').str[0] + ' ' + df['Hora'].astype(str), format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    for col in ['Temperatura Externa', 'Umidade do Ar Externa', 'Pluviômetro (Chuva ) mm', 'Radiação Solar (Wm2)']:
        if col in df.columns: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    return df.sort_values('timestamp')

try:
    df = carregar_dados()
    if df is None:
        st.error("Arquivo de dados não encontrado!")
        st.stop()

    # Carregar Bases de Referência de Risco
    if os.path.exists('referencia_safras.csv') and os.path.exists('regras_pragas.csv'):
        df_ref_safra = pd.read_csv('referencia_safras.csv')
        df_ref_pragas = pd.read_csv('regras_pragas.csv')
    else:
        st.warning("Arquivos de referência (referencia_safras.csv / regras_pragas.csv) não encontrados.")
        st.stop()

    # 2. Barra Lateral
    st.sidebar.header(" Filtros")
    data_min, data_max = df['timestamp'].min().date(), df['timestamp'].max().date()
    selecao = st.sidebar.date_input("Período de Análise", value=[data_min, data_max], min_value=data_min, max_value=data_max)
    btn_previsao = st.sidebar.button(" Gerar Previsão IA (6h)")

    # 3. Diagnóstico do Período
    if isinstance(selecao, (list, tuple)) and len(selecao) == 2:
        inicio, fim = selecao
        df_f = df.loc[(df['timestamp'].dt.date >= inicio) & (df['timestamp'].dt.date <= fim)]

        if not df_f.empty:
            st.subheader(f" Diagnóstico: {inicio.strftime('%d/%m/%Y')} até {fim.strftime('%d/%m/%Y')}")
            c1, c2, c3, c4 = st.columns(4)
            u_med = df_f['Umidade do Ar Externa'].mean()
            
            c1.metric("Máxima Térmica", f"{df_f['Temperatura Externa'].max():.1f} °C")
            c2.metric("Umidade Média", f"{u_med:.1f}%")
            c3.metric("Chuva Acumulada", f"{df_f['Pluviômetro (Chuva ) mm'].sum():.1f} mm")
            c4.metric("Radiação Solar Máx", f"{df_f['Radiação Solar (Wm2)'].max():.0f} W/m²")

            # 4. Gráficos
            tab1, tab2 = st.tabs([" Clima", " Chuvas"])
            with tab1:
                st.plotly_chart(px.line(df_f, x='timestamp', y=['Temperatura Externa', 'Umidade do Ar Externa'], title="Histórico Climático"), use_container_width=True)
            with tab2:
                st.plotly_chart(px.bar(df_f, x='timestamp', y='Pluviômetro (Chuva ) mm', title="Precipitação (mm)"), use_container_width=True)

            # 5. INTELIGÊNCIA DE SAFRA E RISCO DE PREJUÍZO
            st.markdown("---")
            st.subheader(" Análise de Risco de Prejuízo e Pragas (Campo Verde)")
            
            mes_analise = inicio.month
            culturas = ["Soja", "Milho", "Algodão"]
            cols_c = st.columns(3)

            for i, cultura in enumerate(culturas):
                with cols_c[i]:
                    st.markdown(f"### {cultura}")
                    
                    # Risco de Produtividade (Janela)
                    info_s = df_ref_safra[(df_ref_safra['Cultura'] == cultura) & (df_ref_safra['Mes'] == mes_analise)]
                    if not info_s.empty:
                        risco_val = info_s.iloc[0]['Risco_Produtividade']
                        if risco_val == "Baixo": st.success(f"Janela: {info_s.iloc[0]['Status']}")
                        elif risco_val == "Médio": st.warning(f"Risco de Perda: {risco_val}")
                        else: st.error(f"Risco de Perda: {risco_val}")
                        st.caption(f"ℹ {info_s.iloc[0]['Observacao']}")
                    
                    # Risco de Pragas (Baseado nos dados da Estação)
                    st.markdown("** Alerta Fitossanitário:**")
                    pragas = df_ref_pragas[(df_ref_pragas['Cultura'] == cultura) & (u_med >= df_ref_pragas['Umidade_Critica'])]
                    if not pragas.empty:
                        for _, p in pragas.iterrows():
                            st.warning(f"**{p['Praga_Doenca']}**: {p['Alerta']}")
                    else:
                        st.success("Baixa pressão de pragas críticas.")

        # 6. Previsão IA
        if btn_previsao:
            st.markdown("---")
            if os.path.exists('modelo_clima_ifmt.pkl'):
                modelo = joblib.load('modelo_clima_ifmt.pkl')
                ultimo = df.iloc[-1]
                st.subheader(" Previsão IA para as Próximas 6 Horas")
                prevs = []
                t_base, u_base = ultimo['Temperatura Externa'], ultimo['Umidade do Ar Externa']
                for h in range(1, 7):
                    # Ajuste as colunas abaixo conforme o treinamento do seu modelo original
                    X = pd.DataFrame([[t_base, u_base, t_base, u_base, 400]], columns=['Temperatura Externa', 'Umidade do Ar Externa', 'temp_anterior', 'umid_anterior', 'rad_anterior'])
                    t_pred = modelo.predict(X)[0]
                    prevs.append({"Horário": (ultimo['timestamp'] + pd.Timedelta(hours=h)).strftime('%H:%M'), "Temp. Prevista": f"{t_pred:.1f} °C", "Umid. Est.": f"{u_base:.0f}%"})
                    t_base = t_pred
                st.table(pd.DataFrame(prevs))
            else:
                st.error("Modelo IA (.pkl) não encontrado.")

except Exception as e:
    st.error(f"Erro inesperado: {e}")
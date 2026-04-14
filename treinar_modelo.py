import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def treinar_previsao_climatica():
    # 1. Carregar dados validados
    df = pd.read_csv('dados_estacao_limpos.csv', low_memory=False)
    df = df[~df['Data'].str.contains('Date|Data', na=False)]
    df.columns = [col.replace('\n', ' ').strip() for col in df.columns]
    
    # Converter para numérico
    df['Temperatura Externa'] = pd.to_numeric(df['Temperatura Externa'], errors='coerce')
    df['Umidade do Ar Externa'] = pd.to_numeric(df['Umidade do Ar Externa'], errors='coerce')
    df['Radiação Solar (Wm2)'] = pd.to_numeric(df['Radiação Solar (Wm2)'], errors='coerce')
    df = df.dropna(subset=['Temperatura Externa'])

    # 2. FEATURE ENGINEERING (O "Cérebro" do Algoritmo)
    # Criamos colunas que mostram o que aconteceu 1 hora atrás (Lag)
    df['temp_anterior'] = df['Temperatura Externa'].shift(1)
    df['umid_anterior'] = df['Umidade do Ar Externa'].shift(1)
    df['rad_anterior'] = df['Radiação Solar (Wm2)'].shift(1)
    
    # Alvo: Prever a temperatura da PRÓXIMA hora
    df['target'] = df['Temperatura Externa'].shift(-1)
    
    df = df.dropna()

    # 3. Seleção de Variáveis e Alvo
    X = df[['Temperatura Externa', 'Umidade do Ar Externa', 'temp_anterior', 'umid_anterior', 'rad_anterior']]
    y = df['target']

    # Treino e Teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. MODELO: Gradient Boosting (Excelente para séries temporais)
    modelo = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)

    # 5. Validação
    previsoes = modelo.predict(X_test)
    erro = mean_absolute_error(y_test, previsoes)
    print(f"✅ Modelo treinado! Erro médio: {erro:.2f}°C")

    # 6. Salvar o modelo para usar no Dashboard
    joblib.dump(modelo, 'modelo_clima_ifmt.pkl')
    print("💾 Modelo salvo como 'modelo_clima_ifmt.pkl'")

if __name__ == "__main__":
    treinar_previsao_climatica()
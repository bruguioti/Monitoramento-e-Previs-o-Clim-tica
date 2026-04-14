import pandas as pd
import os


arquivo_entrada = 'dados.xlsx'
arquivo_saida = 'dados_estacao_limpos.csv'

def iniciar_conversao():
    if not os.path.exists(arquivo_entrada):
        print(f"O arquivo '{arquivo_entrada}' não foi encontrado.")
        return

    print(" Lendo Excel e convertendo para CSV... Por favor, aguarde.")
    
    try:
       
        df = pd.read_excel(arquivo_entrada, engine='openpyxl')

    
        df = df.dropna(how='all', axis=0)

   
        df.to_csv(arquivo_saida, index=False, encoding='utf-8-sig')
        
        print(f" Gerado: {arquivo_saida}")
        print(f"Foram processadas {len(df)} linhas.")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    iniciar_conversao()
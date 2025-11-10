import joblib
import numpy as np

model = joblib.load("filme_model.pkl")

novo_filme = np.array([[130, 25000000, 85]])
previsao = model.predict(novo_filme)
print(f"Nota prevista: {previsao[0]:.2f}")

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("data/filmes.csv")

X = data[["duracao", "orcamento", "popularidade"]]
y = data["nota"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "filme_model.pkl")
print("Modelo treinado e salvo!")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#importer les donnees
dataset = pd.read_csv(r"C:\Users\HAB SOLUTIONS\OneDrive\Documents\s6\Analyse des donnees\Automobile_data.csv",sep=";")
print(dataset)
# Centrer et réduire les données
Xcr=(dataset-dataset.mean())/dataset.std()
print(Xcr)
# Calculer les valeurs et les vecteurs propres
matrice= Xcr.to_numpy()
matrice_correlation = np.corrcoef(matrice, rowvar=False)
valeurs_propres, vecteurs_propres = np.linalg.eigh(matrice_correlation)
# Afficher les valeurs et les vecteurs propres
print("Valeurs propres: ", valeurs_propres)
print("Vecteurs propres: ", vecteurs_propres)
#creation de la nouvelle matrice
matrice_resultante = Xcr.dot(vecteurs_propres)
#le nuage des points
pd.plotting.scatter_matrix(matrice_resultante, alpha=0.5, figsize=(8, 8), diagonal='hist')
plt.show()
# Calculer la variance expliquée par chaque composante principale
variance_expliquee = valeurs_propres / np.sum(valeurs_propres)
#dessiner le cercle de corelation
# Récupérer les noms des variables initiales
variables = list(Xcr.columns)

# Créer la figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

# Dessiner le cercle de corrélation
for i in range(len(variables)):
    ax.arrow(0, 0, vecteurs_propres[i, 0], vecteurs_propres[i, 1], head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.text(vecteurs_propres[i, 0]* 1.2, vecteurs_propres[i, 1] * 1.2, variables[i], color='k')

# Définir les limites des axes
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Afficher le cercle de corrélation
ax.set_aspect('equal', adjustable='box')
plt.show()

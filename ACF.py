#les bibliotheques

import pandas as pd
import numpy as np
import itertools
from scipy . stats import chi2_contingency
import seaborn as sns ;
sns.set()
import matplotlib . pyplot as plt

#les variables qualitatives

hair = ["chatains","Roux","Blonds"]
eyes = ["Marrons","Noisettes","Verts","Bleus"]

#lire les donnees a partir du fichier excel( que j'ai deja cree)

data = pd.read_csv(r"C:\Users\HAB SOLUTIONS\OneDrive\Documents\s6\Analyse des donnees\accf.csv",sep=";")

#convertir a numpay array

finalData = data.to_numpy()

#calculer la somme des effectifs du tableau de données
sommeTotal = np.sum (finalData)

#diviser la matrice des effectifs par la somme total pour avoir la matrice de correspondance
correspondenceMatrix = np.divide ( finalData , sommeTotal )

#creer la matrice d'independance qui est le produit de la somme marginale des lignes (vecteur colonne) par la somme marginale de colonnes (vecteur ligne)
#somme maginale des lignes
rowTotals = np.sum ( correspondenceMatrix , axis =1)
#somme maginale des colonnes
columnTotals = np . sum ( correspondenceMatrix , axis =0)
#maintenant on calcule la matrice d'independance
independenceMatrix = np.outer ( rowTotals , columnTotals )

#calculer la Matrice d’écart à l’indépendance
ecartIndep = np.divide(( correspondenceMatrix -independenceMatrix ) , np.sqrt( independenceMatrix ) )

#decomposition la Matrice d’écart à l’indépendance en matrice des vecteurs propres
RowMatrix,ValPropresMatrix , ColumnMatix = np . linalg . svd ( ecartIndep , full_matrices =False )
print("matrice des vecteurs propres lignes",RowMatrix)
print("matrice des vecteurs propres colonnes",ColumnMatix)
print("matrice des valeurs propres",ValPropresMatrix)
#obtension des valeurs propres a partir de ces matrices
#valeurs propres lignes
VpLigne = np.zeros((RowMatrix.shape [0] , RowMatrix.shape [1]))
for i in range (RowMatrix.shape[0]) :
    VpLigne[i] = np.divide( RowMatrix[i] , np.sqrt( rowTotals[i]))
#valeurs propres colonnes
VpCol = np.zeros((ColumnMatix.shape [0] , ColumnMatix.shape[1]))
for i in range(ColumnMatix.shape[0]):
    VpCol[i] = np.divide(ColumnMatix[i], np.sqrt(columnTotals[i]))
print("les valeurs propres lignes",VpLigne)
print("les valeurs propres colonnes",VpCol)
#calculer les coordonnées des lignes et des colonnes
rowCoordinates = np.dot ( VpLigne , np.diag ( ValPropresMatrix) )
print("les coordonnees lignes",rowCoordinates)
colCoordinates = np.dot ( VpCol , np.diag ( ValPropresMatrix ) )
print("les coordonnees colonnes",colCoordinates)
#l'interpretation graphique
# afficher le nuage de points avec les étiquettes "hair" et "eyes"
fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(rowCoordinates)):
    ax.scatter(rowCoordinates[i, 0], rowCoordinates[i, 1], color='blue')
    ax.text(rowCoordinates[i, 0] + 0.005, rowCoordinates[i, 1] + 0.005, f'{hair[i]} - {eyes[i]}', fontsize=8)

# ajouter les titres et les étiquettes d'axes
ax.set_title("Nuage de points avec les variables qualitatives hair et eyes")
ax.set_xlabel("Composante principale 1")
ax.set_ylabel("Composante principale 2")

# afficher la figure
plt.show()

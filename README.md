
# Kaggle Killa

Framework permettant une approche industrielle et systématique d'un problème *"Machine Learning"*.

<img src="img/kagglekillawhite.png" width="900">

# Motivation

Lors de la première compétition Kaggle à laquelle nous avons participé, nous avons crée 137 modèles à partir de 24 notebooks, et ce malgré la relative simplicité du problème. Parvenir à développer un modèle performant nécessite une répétition considérable de petites tâches : créer des features, encoder des features, choisir un modèle, tunner les paramètres, appliquer le modèle sur le test set, sauver les prédictions, les stacker, envoyer une soumission, puis recommencer. Nous avons voulu développer une approche générique qui pourrait s'appliquer à n'importe quel problème de machine learning. Ainsi avec seulement 4 notebooks, notre approche permet : 
 1. Le **Features Engineering** et le **Features Encoding** appliqué au train et au test set
 2. La **Multi Modélisation** et le **Tracking des résultats** (le fitting de plusieurs modèles avec plusieurs sets d'hyperparamètres en une seule cellule)
 3. Un **modèle de Stacking** basé sur les résultats des modélisations
 4. La **Multi Soumission** à Kaggle (envoyer plusieurs soumissions en une seule cellule)
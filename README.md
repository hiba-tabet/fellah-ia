# Fellah-IA

Application hors-ligne (Edge-AI) qui détecte les maladies de la tomate, de la pomme de terre et de l'olivier à partir d'une photo de feuille, et donne le conseil de traitement en Darija (texte et audio).

## Principe

Le modèle tourne directement dans le navigateur/téléphone avec ONNX, sans serveur de diagnostic.

## Modèle

MobileNetV3, 16 classes. Export : `export/model.onnx`, `classes.json`, `preprocessing.json`.

## Résultats mesurés

| Test | Résultat |
|---|---|
| PlantVillage + Olive, laboratoire (3 721 images) | 98,2 % |
| PlantDoc, photos réelles (85 images, petit échantillon) | 49,4 % |
| Zeytin, test binaire spécifique (286 images, v4) | 99,1 % équilibré* |

\* Résultat expérimental sur ce jeu de données précis, à ne pas interpréter comme une précision générale sur les oliviers en conditions réelles. Un contrôle de similarité des images est encore en cours.

L'écart entre les images de laboratoire et les photos réelles reste important : le projet le mesure et cherche à le réduire avec des données réelles et des seuils de confiance.

## Données

PlantVillage (Hughes et Salathé, 2015), PlantDoc (Singh et al., 2020, CC BY 4.0), Olive Leaf Image Dataset (Kaggle), Zeytin (Kaggle).

Les licences et conditions d'utilisation doivent être vérifiées sur les pages originales de chaque dataset.

## Organisation

`scripts/` : préparation, entraînement, évaluation et export  
`export/` : modèle ONNX et fichiers nécessaires à l'inférence  
`docs/results/` : résultats expérimentaux  
`app/` : application

# Intégration du modèle Fellah-IA

## Fichiers

- public/model/model.onnx : modèle IA
- public/model/classes.json : liste des 16 classes
- public/model/preprocessing.json : paramètres de prétraitement
- src/inference.js : moteur d'inférence

## Utilisation

Le frontend envoie une photo à la fonction `analyze()`.

```js
import { analyze } from "./inference.js";

const result = await analyze(imageFile);
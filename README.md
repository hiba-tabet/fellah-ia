# fellah AI

fellah AI is an agricultural AI application designed to help farmers identify common plant diseases from leaf images and understand the result through simple Algerian Darija audio guidance.

The application combines an AI image classification model, a structured agricultural knowledge base, and an accessible voice-based interface designed for practical use in the field.

## Target Users

fellah AI was designed specifically to be accessible to farmers who may have limited time, limited technical experience, or limited literacy.

For this reason, the application is intentionally simple:

**Take a photo → Analyze the leaf → Receive the result → Listen to the explanation**

The goal is to reduce the need for reading, typing, and complex interactions.

## Why Algerian Darija?

The application uses Algerian Darija because it is the everyday language spoken by many Algerian farmers.

Technical agricultural information can be difficult to understand when presented only in formal or highly technical language. Using Algerian Darija makes the result more direct, familiar, and practical.

The application interface and audio guidance are therefore designed around the local context rather than a generic international agricultural interface.

## Audio-Based Accessibility

Audio is an important part of fellah AI.

Instead of requiring the farmer to read long explanations, the application provides a dedicated audio explanation for each supported plant condition.

Each supported class has a dedicated audio file in Algerian Darija.

This approach is particularly useful for users with limited literacy or users who prefer listening to instructions while working in the field.

## Supported Plants and Conditions

The current model supports 16 classes covering tomato, potato, and olive plants.

### Tomato

* Tomato bacterial spot
* Tomato early blight
* Tomato late blight
* Tomato leaf mold
* Tomato mosaic virus
* Tomato septoria leaf spot
* Tomato spider mites
* Tomato target spot
* Tomato yellow leaf curl virus
* Tomato healthy

### Potato

* Potato early blight
* Potato late blight
* Potato healthy

### Olive

* Olive peacock spot
* Olive aculus olearius
* Olive healthy

## How the Application Works

The main workflow is:

1. The user selects a crop.
2. The user takes a photo or selects an image from the gallery.
3. The image is resized and preprocessed in the browser.
4. The ONNX model analyzes the image locally.
5. The application returns the most likely class and confidence information.
6. The corresponding agricultural knowledge entry is displayed.
7. The user can listen to the associated Algerian Darija audio.
8. The result can be stored in the local history.

The application is designed to perform inference locally in the browser rather than sending plant images to a remote inference API.

## AI Model

The application uses an ONNX image classification model executed with ONNX Runtime Web.

The model expects:

* RGB image input
* 224 × 224 resolution
* ImageNet-style normalization
* mean: `[0.485, 0.456, 0.406]`
* standard deviation: `[0.229, 0.224, 0.225]`

The browser inference pipeline uses the WebAssembly execution provider.

The application uses a confidence threshold of **0.55** for the current V1 behavior. Predictions below this threshold are treated as uncertain rather than presented as reliable diagnoses.

## Model Evaluation

The model was evaluated using both internal test data and external PlantDoc data.

### Internal Test Set

The final internal test set contains **3,721 images**.

* Top-1 accuracy: **98.74%**
* Macro F1-score: approximately **0.988**
* Weighted F1-score: approximately **0.987**

These results represent the model's performance on the final internal evaluation split.

### PlantDoc External Evaluation

PlantDoc was used as an external evaluation source to measure generalization to images outside the internal training distribution.

A broader PlantDoc evaluation contained:

* **967 images**
* **11 mapped classes**
* Top-1 accuracy: **23.3%**
* Top-3 accuracy: **48.5%**

A separate 85-image PlantDoc subset was also evaluated for the confidence-threshold analysis.

At the selected threshold of **0.55**:

* Coverage: **49.41%**
* Precision among accepted predictions: **61.90%**
* Refusal rate: **50.59%**

This highlights an important limitation: performance on controlled/internal data is substantially higher than performance on external field-style imagery.

## Olive Evaluation

A separate olive evaluation was performed using a balanced dataset.

For the diseased olive subset:

* Culture recognition: **97.6%**
* Healthy/diseased correctness: **99.2%**
* Balanced healthy/diseased precision: **92.4%**

The olive results indicate strong performance for the specific evaluation setup, while still requiring additional field validation before being considered a clinical or agricultural diagnostic system.

## Knowledge Base

The application uses a structured JSON knowledge base stored at:

`public/knowledge.json`

The knowledge base contains one entry for each supported model class.

Each entry provides information such as:

* disease or healthy status
* human-readable title
* description
* class identifier

The frontend maps the predicted model class to the corresponding knowledge entry before displaying the final result.

## Audio Files

Audio files are stored in:

`public/audio/`

There are currently 16 audio files, one for each supported model class.

The recordings are designed to provide simple Algerian Darija explanations associated with the detected condition.

Audio is loaded locally by the application and can also be cached by the service worker after being requested.

## History

The application provides a local history feature.

Detected results can be stored in browser local storage under:

`fellah-ai-history`

The history interface allows the user to:

* review previous analyses
* open previous results
* delete individual results
* delete the complete history

The history is stored locally in the browser.

## PWA and Offline Support

fellah AI is implemented as a Progressive Web App.

The PWA configuration is provided through:

* `public/manifest.webmanifest`
* `public/sw.js`

The service worker caches the main application assets, model files, knowledge base, and ONNX Runtime WebAssembly runtime.

Requested resources such as audio files are also cached dynamically.

### Offline Validation

Offline functionality was tested on the production-style local deployment.

The validation procedure was:

1. Open the application while connected.
2. Perform a real image analysis.
3. Play the corresponding audio.
4. Disable the network connection.
5. Perform another analysis without reloading the application.
6. Verify that the result and audio remain available.

The tested flow worked successfully without an active network connection.

## Project Structure

The main project structure is:

```text
fellah-ia/
├── public/
│   ├── audio/
│   ├── icons/
│   ├── model/
│   │   ├── model.onnx
│   │   ├── classes.json
│   │   └── preprocessing.json
│   ├── knowledge.json
│   ├── manifest.webmanifest
│   ├── ort-wasm-simd-threaded.mjs
│   ├── ort-wasm-simd-threaded.wasm
│   └── sw.js
├── src/
│   ├── inference.ts
│   └── routes/
│       └── index.tsx
├── data/
├── docs/
├── scripts/
├── package.json
├── wrangler.json
└── README.md
```

## Technologies

The project uses:

* TypeScript
* React
* TanStack Start
* Vite
* ONNX Runtime Web
* ONNX
* WebAssembly
* Cloudflare Workers / Wrangler
* Progressive Web App technologies
* Browser Local Storage
* HTML Media APIs
* Camera APIs

## Datasets

The model development and evaluation process used several datasets, including:

* PlantVillage
* PlantDoc
* Zeytin / olive leaf data

The project keeps large datasets and training artifacts outside the deployed frontend repository.

Dataset licenses and original dataset terms remain applicable to their respective sources.

## Limitations

fellah AI is a prototype and research-oriented agricultural assistance tool.

It should not be considered a definitive agricultural diagnosis system.

Important limitations include:

* performance decreases on external field-style images compared with the internal test set
* PlantDoc results show significant domain-shift effects
* image quality strongly affects prediction reliability
* some visually similar diseases remain difficult to distinguish
* the current model only supports the listed 16 classes
* confidence thresholding can reject many uncertain images
* agricultural recommendations should be validated by qualified agricultural professionals when necessary

The system should therefore be used as a decision-support and educational tool rather than as a replacement for professional agricultural expertise.

## Future Work

Possible future improvements include:

* collecting more Algerian field images
* expanding the number of supported crops
* improving external-domain generalization
* adding more field-specific disease classes
* improving uncertainty detection
* adding richer agricultural recommendations
* improving the unclear-image audio guidance
* packaging the PWA as an Android application
* conducting larger real-world field evaluations

## Development

Install dependencies:

```bash
npm install
```

Build the application:

```bash
npm run build
```

For a production-style local test with Wrangler:

```bash
npx wrangler dev .output/server/index.mjs --local --port 8080
```

The application can then be tested locally through the Wrangler server.

## Conclusion

fellah AI combines computer vision, local browser inference, Algerian Darija, and audio accessibility to provide a practical agricultural assistance prototype.

The project demonstrates that an AI-based plant disease classifier can be integrated into a simple, locally adapted interface designed around the needs of Algerian farmers.

The current evaluation also highlights an important research challenge: strong performance on internal data does not automatically translate into strong performance on real-world field imagery.

For that reason, future development will focus on broader field data, stronger external validation, and improved uncertainty handling.


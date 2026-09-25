# fellah AI

fellah AI is an agricultural AI application designed to help farmers identify common plant diseases from a leaf photo and receive clear, practical advice in Algerian Darija.

The application combines an AI image classification model, a knowledge base, and audio explanations to provide an accessible agricultural assistance tool.

## Target Users

fellah AI was designed specifically to be accessible to farmers who may have limited or no reading and writing skills, as well as limited experience with smartphones and digital applications.

For this reason, the application is intentionally simple:

**Take a photo → Analyze the leaf → Receive the result → Listen to the explanation and advice**

The goal is to reduce the need for reading, typing, and complex interaction with technology.

## Why Algerian Darija?

The application uses Algerian Darija because it is the everyday language used by many of the target users.

Technical agricultural information can be difficult to understand when presented only in formal or technical language. Providing explanations in Algerian Darija makes the information more familiar and accessible.

## Audio-Based Accessibility

Audio is an important part of fellah AI.

Instead of requiring the farmer to read long explanations, the application can provide an audio explanation corresponding to the detected plant condition.

Each supported class has a dedicated audio file.

This approach is particularly useful for users with limited literacy or limited experience with digital applications.

## Supported Plants and Conditions

The current model contains 16 classes covering olive, potato, and tomato plants.

### Olive

* Olive healthy
* Olive peacock spot
* Olive aculus olearius

### Potato

* Potato healthy
* Potato early blight
* Potato late blight

### Tomato

* Tomato healthy
* Tomato bacterial spot
* Tomato early blight
* Tomato late blight
* Tomato leaf mold
* Tomato mosaic virus
* Tomato septoria leaf spot
* Tomato spider mites
* Tomato target spot
* Tomato yellow leaf curl virus

## AI Model

The application uses an ONNX image classification model for plant disease recognition.

The model receives a leaf image and produces probabilities for the supported classes.

The application then ranks the predicted classes and uses the highest-confidence prediction as the main result.

## Model Files

The model files used by the application are stored in:

```text
app/public/model/
```

Current model assets include:

```text
app/public/model/
├── model.onnx
├── classes.json
└── preprocessing.json
```

### `model.onnx`

The trained ONNX model used for image classification.

### `classes.json`

Contains the list of classes predicted by the model.

### `preprocessing.json`

Contains the preprocessing information required before sending an image to the model.

## Knowledge Base

The agricultural explanations and recommendations are stored in:

```text
data/knowledge.json
```

The knowledge base contains information corresponding to all 16 model classes.

It is used to provide additional information after the model produces a prediction.

The knowledge base is designed to connect the AI prediction with practical agricultural information.

## Audio Files

Audio explanations are stored in:

```text
audio/
```

There are currently 16 audio files, corresponding to the 16 supported classes.

Each audio file uses the same class identifier as the model and knowledge base.

This makes it possible to connect:

```text
Prediction → Knowledge → Audio explanation
```

## Model Evaluation

The model was evaluated using test images from PlantDoc.

The evaluation focused on the relationship between prediction confidence, coverage, precision, and refusal rate.

A confidence threshold of `0.55` is currently used by the application.

At this threshold, the evaluation produced:

| Confidence Threshold | Coverage | Precision | Refusal Rate |
| -------------------: | -------: | --------: | -----------: |
|                 0.55 |   49.41% |    61.90% |       50.59% |

The threshold was selected as a practical V1 compromise between providing predictions and refusing uncertain predictions.

The threshold can be adjusted in the future as more representative test data becomes available.

## Confidence-Based Prediction

The application does not treat every model prediction as equally reliable.

A prediction is considered confident when its probability reaches the configured confidence threshold.

The current value is:

```javascript
const CONFIDENCE_THRESHOLD = 0.55;
```

If the highest prediction does not reach this threshold, the application can treat the result as uncertain rather than presenting it as a reliable diagnosis.

This is important because an agricultural AI system should avoid presenting uncertain predictions as definitive diagnoses.

## Project Structure

The main project structure is currently organized as follows:

```text
fellah-ia/
├── app/
│   ├── public/
│   │   └── model/
│   │       ├── model.onnx
│   │       ├── classes.json
│   │       └── preprocessing.json
│   ├── src/
│   │   └── inference.js
│   └── INTEGRATION.md
│
├── audio/
│   └── *.mp3
│
├── data/
│   └── knowledge.json
│
├── docs/
│
├── export/
│   └── classes.json
│
├── models/
│
├── scripts/
│   └── choose_threshold.py
│
├── package.json
├── README.md
└── sw.js
```

Some directories also contain datasets, evaluation files, experiments, and training-related material used during development.

## Technologies

The project currently uses technologies including:

* JavaScript
* React
* Vite
* ONNX Runtime
* ONNX image classification
* Python
* Plant disease datasets
* Audio explanations in Algerian Darija

## Development

The project uses Node.js and npm for the frontend environment.

Install the dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Build the application:

```bash
npm run build
```

Create a development build:

```bash
npm run build:dev
```

Preview a production build:

```bash
npm run preview
```

Run linting:

```bash
npm run lint
```

Format the project:

```bash
npm run format
```

## Offline and PWA Support

Offline functionality and Progressive Web App support are part of the project's objectives.

The application is intended to eventually allow the main user experience to work with limited or no internet connectivity.

This includes making the following resources available offline:

* The application interface
* The ONNX model
* Class definitions
* Preprocessing configuration
* Agricultural knowledge
* Audio explanations

The final service worker and PWA configuration will be integrated once the frontend structure is finalized.

## Accessibility Goals

fellah AI is designed around several accessibility principles:

### Minimal Interaction

The farmer should not need to navigate through complicated menus.

### Photo-Based Input

The main input is a photograph of a plant leaf.

### Audio Assistance

Important information can be communicated through audio instead of relying only on text.

### Familiar Language

Agricultural explanations are provided in Algerian Darija.

### Simple User Flow

The intended interaction is:

```text
Take a photo
      ↓
Analyze the leaf
      ↓
Receive the prediction
      ↓
Listen to the explanation
      ↓
Follow the recommended advice
```

## Project Objectives

The main objectives of fellah AI are:

1. Help farmers identify common plant diseases from photographs.
2. Provide accessible agricultural information.
3. Reduce dependence on reading and writing.
4. Use Algerian Darija for practical explanations.
5. Provide audio assistance.
6. Explore offline and PWA functionality.
7. Provide a simple interface suitable for users with limited experience with smartphones.

## Limitations

The application is an AI-assisted agricultural tool and should not be considered a replacement for professional agronomic diagnosis.

The model is trained and evaluated on specific datasets, and its performance may vary with:

* Image quality
* Lighting conditions
* Leaf orientation
* Background complexity
* Plant variety
* Disease stage
* Diseases or conditions not represented in the training data

The current confidence threshold also means that many uncertain images may be rejected rather than classified.

The evaluation reported above is based on a limited test set and should not be interpreted as a guarantee of real-world performance.

## Future Improvements

Possible future improvements include:

* Improving model accuracy with more representative field images.
* Expanding the number of supported crops and diseases.
* Improving disease-specific recommendations.
* Improving audio explanations.
* Adding more robust offline functionality.
* Completing the PWA implementation.
* Adding application icons and installation support.
* Improving image quality checks.
* Adding more extensive field testing.
* Improving confidence calibration.
* Adding additional languages if needed.

## Current Project Status

The project currently includes:

* A trained ONNX plant classification model.
* 16 supported classes.
* Model inference logic.
* A confidence threshold of `0.55`.
* A knowledge base covering all 16 classes.
* 16 Algerian Darija audio explanations.
* Model assets prepared for frontend integration.
* Evaluation and threshold-selection scripts.
* GitHub version control.

The frontend is still being finalized and integrated.

The service worker, PWA configuration, final application flow testing, and demonstration workflow will be completed after the frontend structure is finalized.

## Vision

fellah AI aims to make agricultural AI more accessible by adapting the technology to the real needs of its users.

The project focuses not only on disease classification, but also on accessibility, language, audio communication, simplicity, and practical agricultural assistance.

The long-term goal is to provide farmers with a tool that is easy to understand and easy to use, even when reading, writing, smartphone experience, or internet connectivity may be limited.

## Repository

GitHub repository:

`https://github.com/hiba-tabet/fellah-ia`

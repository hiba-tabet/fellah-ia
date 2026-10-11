# Fellah AI
An accessible, privacy-first agricultural assistance tool for Algerian farmers. Take a photo. Understand the result. Listen to practical guidance in Algerian Darija.


Live demo: https://fellah-ai.karachorouk07.workers.dev
Source code: github.com/hiba-tabet/fellah-ia
Inference: In the browser
## Overview
Fellah AI is a research-oriented agricultural application that helps users identify selected plant diseases and healthy-plant conditions from a photograph of a leaf. It combines:

- An ONNX image-classification model executed locally in the browser
- A structured agricultural knowledge base
- Audio explanations in Algerian Darija
- A simple, camera-first user experience
- Local analysis history stored on the user's device
- Progressive Web App features for an installable, field-oriented experience

The project was designed around a practical accessibility problem: useful agricultural information is often difficult to access when a user has limited literacy, limited time, limited smartphone experience, or limited access to specialist advice. Fellah AI therefore minimizes typing and reading and focuses on a simple flow:
Choose a crop → Take or select a photo → Analyze the leaf → Listen to the explanation

## Live application
Open the deployed Fellah AI demo
The current deployment runs on Cloudflare Workers. The application does not require a separate inference server for its core image-analysis workflow.
## Project goals
- Make basic plant-health information more accessible to Algerian farmers.
- Reduce reliance on reading, typing, and complex navigation.
- Provide explanations in a familiar local language: Algerian Darija.
- Keep image inference local to the user's browser where possible.
- Present uncertainty honestly instead of treating every prediction as a definitive diagnosis.
- Create a foundation for future field validation, multilingual support, and broader crop coverage.
## Accessibility and user-centered design
Fellah AI was designed for real-world agricultural use rather than for a technically experienced audience.
### Key design decisions

- Photo-based input: users can take a picture with the phone camera or choose an existing image.
- Minimal interaction: the primary workflow is intentionally short and visual.
- Algerian Darija: explanations are written and recorded for the local target audience.
- Audio guidance: users can press an audio button to hear the result and associated guidance.
- Clear uncertainty state: unclear or low-confidence images are not presented as reliable diagnoses.
- Mobile-first interface: the main interaction is designed around phones used in the field.
- Local history: recent results can be reviewed on the same device without creating an account.

## Supported crops and conditions
The current model contains 16 classes across three crops.
### Olive

- Healthy olive leaf
- Olive peacock spot
- Aculus olearius

### Potato

- Healthy potato leaf
- Potato early blight
- Potato late blight

### Tomato

- Healthy tomato leaf
- Tomato bacterial spot
- Tomato early blight
- Tomato late blight
- Tomato leaf mold
- Tomato mosaic virus
- Tomato septoria leaf spot
- Tomato spider mites
- Tomato target spot
- Tomato yellow leaf curl virus

The class names, agricultural explanations, and audio files are linked through a shared class identifier so that a prediction can be mapped consistently to:
Model prediction → Knowledge entry → Darija explanation → Audio file

## How the AI pipeline works

- The user selects olive, potato, or tomato.
- The user captures or selects a leaf image.
- The browser resizes the image and prepares it for inference.
- ONNX Runtime Web runs the model locally using WebAssembly.
- The model produces ranked class probabilities.
- A confidence threshold is applied.
- The application displays a supported result, or an uncertainty message when the image is not sufficiently clear.
- The corresponding Darija explanation can be played as audio.
- The result may be saved in the browser's local history.

## Model input
The current model expects:

- RGB image input
- 224 × 224 resolution
- ImageNet-style normalization
- Mean: [0.485, 0.456, 0.406]
- Standard deviation: [0.229, 0.224, 0.225]

## Confidence handling
The current application threshold is:
0.55

If the top prediction does not meet this threshold, the application can return an unclear-image result and ask the user to retake the photo. This is a deliberate safety and usability choice: a lower-confidence prediction should not be presented as a certain diagnosis.
## Evaluation and transparency
The project documents both strong internal results and important real-world limitations.
### Internal evaluation
The final internal evaluation reported in the project documentation contains 3,721 images and reports:
- Top-1 accuracy: 98.74%
- Macro F1-score: approximately 0.988
- Weighted F1-score: approximately 0.987
These results describe the internal evaluation setup and should not be interpreted as a guarantee of performance on arbitrary field photographs.
### External PlantDoc evaluation
A broader external PlantDoc evaluation reported:
- 967 images
- 11 mapped classes
- Top-1 accuracy: 23.3%
- Top-3 accuracy: 48.5%
For a separate threshold-analysis subset, the reported results at a confidence threshold of 0.55 were:


These external results show a substantial domain-shift challenge between curated datasets and field-style images. This limitation is central to the responsible interpretation of the project.
### Olive evaluation
The project documentation also reports a separate balanced olive evaluation with:
- Culture recognition: 97.6%
- Healthy/diseased correctness: 99.2%
- Balanced healthy/diseased precision: 92.4%
The evaluation setup, dataset composition, and collection conditions should be considered when interpreting these figures.
## Privacy and data handling
Fellah AI is designed to reduce unnecessary data collection:
- Core image inference runs locally in the browser.
- The application does not require user registration for the main workflow.
- The application does not require a remote image-inference API.
- Analysis history is stored in browser localStorage under fellah-ai-history.
- Images used in the local history remain on the user's device unless the user or browser removes them.
- The application uses camera access only after the user initiates the camera workflow and grants permission.
As with any web application, users should review the hosting provider's policies and their own browser permissions before deploying a production service. The current project is a prototype and does not claim to provide a complete legal privacy policy or enterprise data-governance system.
## Responsible use and limitations
Fellah AI is an agricultural decision-support and educational tool, not a replacement for a qualified agronomist, plant pathologist, extension officer, or other professional advice.
Results may be affected by:
- Poor lighting or blur
- Incorrect framing or a small visible leaf area
- Complex backgrounds
- Leaf orientation
- Crop variety and growth stage
- Disease stage and visual similarity between conditions
- Diseases not represented in the 16 supported classes
- Differences between training images and local field conditions
- Images that do not contain a supported crop or condition
Users should treat an uncertain result as a reason to retake the image or seek expert advice. Recommendations should be validated locally before any treatment, pesticide, or crop-management decision is made.
## Technology stack

- TypeScript
- React 19
- Vite
- TanStack Start and TanStack Router
- ONNX Runtime Web
- ONNX image-classification model
- WebAssembly
- Tailwind CSS
- Cloudflare Workers / Wrangler
- Progressive Web App technologies
- Browser Local Storage and Media APIs

## Repository structure

```text
fellah-ia/
├── public/
│   ├── audio/                  # Algerian Darija audio guidance
│   ├── icons/                  # Application icon assets
│   ├── model/
│   │   ├── model.onnx          # Browser inference model
│   │   ├── classes.json       # Model class definitions
│   │   └── preprocessing.json # Model preprocessing metadata
│   ├── knowledge.json         # Agricultural knowledge entries
│   ├── manifest.webmanifest   # PWA metadata
│   ├── sw.js                  # Service worker
│   └── ort-*.wasm/mjs         # ONNX Runtime Web assets
├── src/
│   ├── inference.ts           # Image preparation and model inference
│   ├── routes/index.tsx       # Main user workflow
│   ├── assets/                # Application images
│   └── lib/                   # Shared utilities and error handling
├── data/                      # Evaluation reports and development data
├── docs/                      # Project documentation and results
├── export/                    # Model export metadata and samples
├── scripts/                   # Training, evaluation, and analysis scripts
├── package.json
├── package-lock.json
├── vite.config.ts
└── README.md
```


## Local development
Requirements
Node.js 22 or a compatible recent Node.js release
npm
A modern browser with camera and WebAssembly support for the full experience
Install dependencies
npm install

Start the development server
npm run dev

Build for production
npm run build

Preview a production build
npm run preview

Lint and format
npm run lint
npm run format

Cloudflare Workers deployment
The project is configured for a Cloudflare-compatible TanStack Start deployment. After authenticating with Wrangler:
npm install
npm run build
npx wrangler login
npx nitro deploy --prebuilt

Alternatively, deploy the generated Worker explicitly:
npx wrangler deploy --config .output/server/wrangler.json

The build generates the Worker configuration and static asset directory under .output/. Do not commit generated build folders such as .output/ or .wrangler/.
## Offline and PWA behavior
The application includes a web manifest and service worker. After the relevant application resources have been loaded, the service worker can cache the application shell, model assets, knowledge base, runtime assets, and requested audio resources.
Offline behavior depends on the browser cache, the first successful online load, device storage, and the specific resource requested. It should therefore be validated on the target devices before being described as guaranteed offline service.
## Data sources and attribution
The project documentation identifies the following development and evaluation sources:
PlantVillage
PlantVillage is a public plant-leaf dataset introduced by Mohanty, Hughes, and Salathé. The project documentation records use of a prepared subset for training.
Project use: training data
Reported original publication: Using Deep Learning for Image-Based Plant Disease Detection
Reported license: CC BY-SA 3.0
PlantDoc
PlantDoc is an external plant-disease dataset containing more field-style imagery than laboratory-only datasets.
Project use: external evaluation and generalization testing
Reported authorship: Singh et al.
Reported license: CC BY 4.0
Olive / Zeytin data
The project documentation describes an olive-leaf dataset containing three classes: healthy, Aculus olearius, and olive peacock spot, with images associated with research conducted in Denizli, Turkey.
The exact licensing terms for this source were not clearly established in the project materials. The project therefore does not claim a specific license for those images and should retain the original source attribution and permissions documentation before redistribution.
Attribution note: Dataset names, publications, image licenses, and redistribution rights should be verified against the original source pages and included in any formal academic, institutional, or public release documentation.
## Future development
Potential next steps include:
Collecting and validating more Algerian field images
Improving calibration and uncertainty detection
Expanding crop and disease coverage
Adding image-quality guidance before inference
Improving agronomic recommendations with local expert review
Expanding Darija audio coverage and adding additional languages
Conducting structured usability testing with farmers
Conducting prospective field validation with agricultural professionals
Improving offline installation and update behavior
Adding an explicit privacy notice and formal governance documentation for production use
## Project status
Fellah AI is an active prototype and research-oriented demonstration. The current repository and deployment include:
A working mobile-first web application
A browser-executed ONNX model
16 supported plant-health classes
Algerian Darija audio guidance
A structured knowledge base
Local analysis history
PWA manifest and service-worker support
Training and evaluation artifacts
A Cloudflare Workers deployment configuration
The project should be evaluated as a practical proof of concept and a foundation for further field research, not as a clinically or agronomically certified diagnostic system.
## Citation and acknowledgement
If you use, present, or extend this project, please acknowledge:
The Fellah AI project and its contributors
The original dataset creators and licensors
The ONNX Runtime and open-source software communities
Agricultural experts and farmers who contribute to future validation
## Contact and repository
Repository: https://github.com/hiba-tabet/fellah-ia
Live demo: https://fellah-ai.karachorouk07.workers.dev
For formal institutional or academic review, include the project maintainer's verified contact information, contributor names, dataset citations, and the applicable source-code and media licenses before publication.

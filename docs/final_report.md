# fellah AI — Final Project Report

## 1. Introduction

Agriculture plays an important role in the daily life and economy of many communities. Plant diseases can significantly affect crop production when they are not identified and managed in time. However, identifying plant diseases from leaf symptoms can be difficult, especially when farmers do not have direct access to agricultural experts or technical information.

**fellah AI** is an agricultural artificial intelligence application designed to assist farmers in identifying common diseases affecting olive, potato, and tomato plants from a photograph of a leaf.

The project combines an image classification model, a structured agricultural knowledge base, and audio explanations in Algerian Darija. The objective is not only to provide an AI prediction, but also to make the information understandable and accessible to users with different levels of literacy and digital experience.

The main interaction is intentionally simple:

**Take a photo → Analyze the leaf → Receive the result → Listen to the explanation and advice**

## 2. Problem and Motivation

Plant disease identification can require agricultural knowledge and experience. In practical situations, farmers may need to recognize symptoms such as spots, discoloration, mold, or other changes on leaves and determine whether these symptoms correspond to a disease.

Access to information can also be a challenge. Technical agricultural resources are often written in formal or specialized language and may assume that the user is comfortable reading, writing, and using digital applications.

fellah AI was therefore designed around two complementary challenges:

1. **Disease identification:** use computer vision to classify a photographed plant leaf into one of the supported classes.
2. **Accessible communication:** present the result and corresponding agricultural information in a form that is easier to understand and use in everyday situations.

The project particularly considers farmers who may have limited or no reading and writing skills, as well as limited experience with smartphones and digital applications.

## 3. Project Objectives

The main objectives of fellah AI are:

* Develop an image classification model capable of recognizing common olive, potato, and tomato leaf conditions.
* Integrate the trained model into a web-based application.
* Provide information associated with each predicted condition through a structured knowledge base.
* Provide audio explanations in Algerian Darija.
* Reduce the dependence on reading, typing, and complex user interaction.
* Explore the feasibility of running the AI inference locally in the application using an ONNX model.
* Prepare the application for offline and Progressive Web App (PWA) functionality.

The final system is intended as an **agricultural assistance and educational prototype**, rather than a replacement for professional agricultural diagnosis.

## 4. System Architecture

fellah AI is organized as a web-based application combining three main components:

1. **AI inference**
2. **Agricultural knowledge base**
3. **Audio assistance**

The general workflow is:

**Leaf photo → Image preprocessing → ONNX model inference → Class ranking → Confidence check → Knowledge lookup → Result and audio explanation**

### 4.1 AI Inference

The AI model is stored as an ONNX file:

```text
public/model/model.onnx
```

The frontend sends the selected image to the `analyze()` function implemented in:

```text
src/inference.ts
```

The inference engine preprocesses the image, runs the ONNX model, applies softmax to the output logits, and ranks the predicted classes.

The highest-confidence class is considered the main prediction.

A confidence threshold is also used to avoid automatically accepting predictions when the model is not sufficiently confident.

### 4.2 Model Input

The model expects images with the following format:

| Parameter     | Value            |
| ------------- | ---------------- |
| Input size    | 224 × 224 pixels |
| Color format  | RGB              |
| Tensor layout | NCHW             |
| Data type     | float32          |
| Resize        | 224 × 224        |
| Cropping      | None             |
| Interpolation | Bilinear         |

Pixel values are first scaled to the `[0, 1]` range and then normalized using the standard ImageNet mean and standard deviation:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

### 4.3 Model Output

The model produces 16 output logits corresponding to the 16 classes defined in `classes.json`.

Softmax is applied to convert these logits into class probabilities.

The predictions are then ranked according to their confidence.

### 4.4 Knowledge Base

After a prediction is obtained, the application can retrieve the corresponding agricultural information from:

```text
public/knowledge.json
```

The knowledge base contains information for all 16 supported classes and connects the AI prediction with practical agricultural information.

### 4.5 Audio Assistance

Each supported class has a corresponding audio explanation.

The project currently contains:

```text
16 audio files
```

one for each of the 16 model classes.

The audio component is designed to reduce dependence on reading and make the application more accessible to users who prefer or require spoken explanations.

### 4.6 Application Integration

The model is designed to run directly in the web application through ONNX inference.

The integration files are:

```text
public/model/model.onnx
public/model/classes.json
public/model/preprocessing.json
src/inference.ts
```

This architecture is intended to support local inference and, once the frontend and offline components are finalized, offline usage without requiring a permanent connection to a remote AI service.


## 5. Datasets and Data Preparation

The model was developed using a combination of public plant disease datasets and an olive leaf dataset. The datasets were selected to cover the main plant species and disease categories supported by fellah AI.

### 5.1 PlantVillage

PlantVillage is a large public plant disease image dataset. The original dataset contains approximately 162,916 image files across different versions, including color, grayscale, and segmented images.

For the fellah AI project, a prepared subset of **10,026 PlantVillage images** was used for model training.

PlantVillage was used as the main source of training images for the supported tomato and potato disease classes.

### 5.2 PlantDoc

PlantDoc was used to improve the evaluation of the model on images that are more representative of real-world conditions.

For the final training process, the PlantDoc data was divided into:

* **750 training images**
* **132 validation images**
* **85 test images**

A broader evaluation was also performed using **967 PlantDoc images** covering 11 classes that could be mapped to the model's supported classes.

The broader evaluation resulted in a Top-1 accuracy of **23.3%** and a Top-3 accuracy of **48.5%**. These results show that performance can decrease significantly when the model is evaluated on a larger and more diverse external dataset.

### 5.3 Olive Leaf Dataset

An olive leaf dataset containing **3,400 images** was used to support the three olive classes:

* Olive healthy
* Aculus olearius
* Olive peacock spot

The original dataset contains:

* 1,050 healthy images
* 890 Aculus olearius images
* 1,460 olive peacock spot images

For the final training process, the olive data was divided into:

* **477 training images**
* **191 validation images**
* **286 test images**

The olive dataset was collected in Denizli, Turkey and was used as an academic research dataset. No specific open license was clearly identified for this dataset, so the project does not claim a specific license for these images.

### 5.4 Data Preprocessing

Before being provided to the model, images are resized to **224 × 224 pixels** without cropping.

The images are converted to RGB format and normalized using the ImageNet mean and standard deviation.

The preprocessing parameters are stored in:

```text
public/model/preprocessing.json
```

This ensures that the preprocessing used by the web application matches the preprocessing expected by the trained model.

### 5.5 Data Preparation and Evaluation

The project distinguishes between training data, validation data, and unseen test data.

The test sets are not used to train the model and are reserved for evaluating its performance.

This separation makes it possible to measure how the model performs on images that were not directly used during training.

Because the datasets come from different sources and have different image characteristics, the evaluation results vary between internal test data and external datasets. These differences are considered an important limitation of the current prototype.

## 6. AI Model and Preprocessing

The fellah AI classification model is designed to recognize **16 plant conditions** across olive, potato, and tomato leaves.

### 6.1 Supported Classes

The model supports the following classes:

**Olive**

* Olive healthy
* Olive Aculus olearius
* Olive peacock spot

**Potato**

* Potato healthy
* Potato early blight
* Potato late blight

**Tomato**

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

### 6.2 Model Input and Preprocessing

The model expects an RGB image with a spatial resolution of **224 × 224 pixels**.

Images are resized directly to the required resolution without cropping. Bilinear interpolation is used during resizing.

Pixel values are divided by 255 to obtain values between 0 and 1. The resulting values are then normalized using the following parameters:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

The resulting tensor uses the **NCHW** layout and `float32` data type.

These preprocessing parameters are stored in:

```text
public/model/preprocessing.json
```

### 6.3 ONNX Model

The trained model was exported to the **ONNX** format so that it can be used directly by the web application.

The model file is:

```text
public/model/model.onnx
```

The ONNX model contains the trained neural network and produces **16 output logits**, one for each supported class.

Softmax is applied to these logits to obtain probability values for the different classes.

### 6.4 Prediction Ranking

After inference, the class probabilities are sorted from the highest confidence to the lowest confidence.

The highest-confidence class is used as the Top-1 prediction.

The inference engine also keeps the ranked predictions, which can be used to provide additional prediction information to the frontend.

The implementation is located in:

```text
src/inference.ts
```

### 6.5 Confidence Threshold

To avoid presenting uncertain predictions as reliable results, the application uses a confidence threshold of:

```text
0.55
```

A prediction is considered sufficiently confident only when its probability reaches or exceeds this threshold.

When the highest probability is below the threshold, the system can treat the result as uncertain rather than presenting it as a confident diagnosis.

This mechanism is particularly important for images that are very different from the training data or that contain ambiguous symptoms.


## 7. Knowledge Base and Audio Assistance

The AI prediction is combined with additional agricultural information to make the result more useful to the user.

### 7.1 Knowledge Base

The agricultural knowledge base is stored in:

```text
public/knowledge.json
```

It contains information for all **16 supported classes**.

For each condition, the knowledge base can provide information such as:

* the name of the condition;
* the affected plant;
* a short description;
* common symptoms;
* recommended actions or advice.

This structure separates the AI prediction from the agricultural information. The model identifies the class, while the knowledge base provides the corresponding explanation and practical information.

### 7.2 Algerian Darija

The application uses **Algerian Darija** for its user-facing audio explanations.

This language choice is intentional. The target users are primarily Algerian farmers, and Darija is commonly used in everyday communication.

Using familiar spoken language makes the information easier to understand and reduces the need for users to interpret technical agricultural terminology.

### 7.3 Audio Assistance

The project contains **16 audio files**, corresponding to the 16 model classes.

Each audio file provides a spoken explanation associated with a specific prediction.

The audio files are stored in:

```text
audio/
```

The audio content is recorded in a voice adapted to the project and is intended to provide accessible explanations and advice without requiring the user to read a long text.

### 7.4 Accessibility Through Audio

Audio is an important accessibility component of fellah AI.

The application is designed for users who may have limited reading and writing skills or limited experience with digital applications.

Instead of requiring the user to read technical information, the application can provide the explanation through spoken Algerian Darija.

This supports the main interaction principle of the project:

**Take a photo → Receive the result → Listen to the explanation and advice**

The combination of AI prediction, agricultural information, and audio assistance is intended to make the application more accessible in real-world agricultural contexts.


## 8. Model Evaluation

The model was evaluated using unseen test data from different sources. The objective was to measure how well the model generalizes beyond the images used during training.

### 8.1 Final Model Results

The final training configuration, referred to as **v4**, used:

* **10,026 PlantVillage images**
* **750 PlantDoc training images**
* **132 PlantDoc validation images**
* **85 PlantDoc test images**
* **477 olive training images**
* **191 olive validation images**
* **286 olive test images**
* **10 training epochs**
* **Apple MPS acceleration during training**

The final model achieved the following results on the available test sets:

| Evaluation set          | Metric                             | Result |
| ----------------------- | ---------------------------------- | -----: |
| PlantDoc test set       | Top-1 accuracy                     |  49.4% |
| Olive test set          | Balanced healthy/diseased accuracy |  99.1% |
| Internal clean test set | Accuracy                           |  98.74% |

These results should be interpreted according to the characteristics of each evaluation set.

### 8.2 PlantDoc Evaluation

The final PlantDoc test split contained **85 images** and resulted in a Top-1 accuracy of **49.4%**.

A separate broader evaluation was performed using **967 PlantDoc images** mapped to 11 supported classes. This evaluation resulted in:

* **Top-1 accuracy: 23.3%**
* **Top-3 accuracy: 48.5%**

The difference between these results is related to the different evaluation sets and protocols. The 85-image test split was the final test subset used during the training pipeline, while the 967-image evaluation provides a broader external evaluation.

The results indicate that the model performs more reliably on some evaluation conditions than on diverse external images.

### 8.3 Olive Dataset Evaluation

The final olive test split contained **286 images**.

The model achieved a balanced healthy/diseased accuracy of **99.1%** on this test split.

A separate evaluation using **954 olive images** was also performed. This evaluation measured the distinction between healthy and diseased olive leaves and obtained a balanced healthy/diseased accuracy of **92.4%**.

These two results correspond to different evaluation sets and should not be considered identical metrics.

### 8.4 Internal Test Evaluation

The final training log reports an accuracy of **98.74%** on an internal clean test set containing **3,721 images**.

This result indicates strong performance on images similar to the data used during the project development.

However, internal test performance should not be interpreted as equivalent to real-world field performance because field photographs can contain different backgrounds, lighting conditions, leaf orientations, image quality, and disease appearances.

### 8.5 Interpretation of the Results

The evaluation demonstrates that the model can recognize several supported plant conditions, but performance varies considerably depending on the evaluation dataset.

The lower performance on the broader PlantDoc evaluation highlights the difficulty of generalizing from curated or training-related images to more diverse real-world images.

For this reason, fellah AI is presented as an **agricultural assistance and educational prototype** rather than a definitive diagnostic system.


## 9. Confidence Threshold Evaluation

A confidence threshold was evaluated to determine when the application should accept a prediction and when it should treat the result as uncertain.

The evaluation was performed on the **85-image PlantDoc test set**.

| Confidence Threshold |   Coverage |  Precision | Refusal Rate |
| -------------------: | ---------: | ---------: | -----------: |
|                 0.00 |    100.00% |     49.41% |        0.00% |
|                 0.30 |     88.24% |     53.33% |       11.76% |
|                 0.40 |     74.12% |     60.32% |       25.88% |
|                 0.50 |     60.00% |     54.90% |       40.00% |
|             **0.55** | **49.41%** | **61.90%** |   **50.59%** |
|                 0.60 |     42.35% |     66.67% |       57.65% |
|                 0.70 |     32.94% |     71.43% |       67.06% |
|                 0.80 |     24.71% |     71.43% |       75.29% |
|                 0.90 |     15.29% |     69.23% |       84.71% |

For the current prototype, a threshold of **0.55** was selected.

At this threshold:

* **Coverage:** 49.41%
* **Precision:** 61.90%
* **Refusal rate:** 50.59%

This means that the application does not attempt to provide a confident prediction for every image. Approximately half of the tested images are rejected when the model confidence is below the selected threshold.

The threshold therefore acts as a safety mechanism against presenting uncertain predictions as reliable results.

The selected value of **0.55** represents a practical prototype configuration. It does not eliminate model errors, and the evaluation results show that further model improvement is required before the system could be considered suitable for critical agricultural diagnosis.


## 10. Limitations

Although fellah AI demonstrates the feasibility of combining image classification, agricultural information, and audio assistance in a single application, the current prototype has several limitations.

### 10.1 Model Generalization

The evaluation results vary significantly between datasets. The model performs strongly on some internal and olive test sets, while performance decreases on the broader PlantDoc evaluation.

This indicates that the model may have difficulty generalizing to images that differ from the training data.

### 10.2 Real-World Image Conditions

Real agricultural photographs can contain challenging conditions such as:

* complex backgrounds;
* poor or variable lighting;
* blurred images;
* different leaf orientations;
* multiple leaves in the same image;
* partially visible leaves;
* symptoms at different stages;
* disease symptoms that look similar.

These conditions may reduce prediction accuracy.

### 10.3 Limited Number of Supported Conditions

The current model supports **16 classes** covering olive, potato, and tomato leaves.

Many other crops and plant diseases are not supported. An unsupported condition may therefore be incorrectly classified or rejected by the confidence mechanism.

### 10.4 Dataset Limitations

The training and evaluation datasets come from different sources and were collected under different conditions.

Some datasets contain relatively controlled images, while external datasets such as PlantDoc contain more diverse images.

The difference between these datasets can affect model performance and makes direct comparison between evaluation results difficult.

### 10.5 Confidence Threshold Limitations

The confidence threshold reduces the number of uncertain predictions presented to the user, but it does not guarantee that an accepted prediction is correct.

At the selected threshold of **0.55**, the evaluation on the 85-image PlantDoc test set produced a precision of **61.90%** among accepted predictions and a refusal rate of **50.59%**.

Therefore, the threshold should be considered a prototype uncertainty mechanism rather than a guarantee of diagnostic reliability.

### 10.6 Prototype Status

fellah AI is currently an **agricultural assistance and educational prototype**.

The application should not be considered a replacement for agricultural experts or laboratory diagnosis. When a plant condition is unclear, severe, or economically important, the result should be verified using appropriate agricultural expertise.

## 11. Conclusion and Future Work

### 11.1 Conclusion

fellah AI demonstrates the feasibility of developing an accessible agricultural assistance application based on artificial intelligence.

The project combines an image classification model, local ONNX inference, a structured agricultural knowledge base, and audio explanations in Algerian Darija.

The application is designed around a simple interaction:

**Take a photo → Analyze the leaf → Receive the result → Listen to the explanation and advice**

This approach is particularly relevant for users who may have limited reading and writing skills or limited experience with smartphones and digital applications.

The evaluation results show that the model can recognize several supported plant conditions, but performance varies depending on the dataset and image conditions. The broader PlantDoc evaluation also demonstrates the difficulty of generalizing to diverse real-world images.

For this reason, the current version is considered an **agricultural assistance and educational prototype**, rather than a definitive plant disease diagnosis system.

### 11.2 Future Work

Several improvements could be developed in future versions of fellah AI:

1. **Improve the AI model**

   * Increase the diversity and quantity of training images.
   * Add more real-world field photographs.
   * Improve performance on external datasets such as PlantDoc.
   * Investigate stronger data augmentation and model architectures.

2. **Expand plant and disease coverage**

   * Add more crops.
   * Add additional diseases and pests.
   * Include more olive, potato, and tomato conditions.

3. **Improve uncertainty detection**

   * Calibrate model confidence more effectively.
   * Investigate alternative confidence and out-of-distribution detection methods.
   * Reduce false confident predictions.

4. **Complete offline and PWA functionality**

   * Finalize the service worker.
   * Add the PWA manifest and application icons.
   * Extend and maintain the validated offline workflow across additional devices and usage scenarios.

5. **Improve accessibility**

   * Continue using Algerian Darija audio.
   * Improve the clarity and consistency of spoken explanations.
   * Minimize the amount of reading and typing required.
   * Keep the interface simple and suitable for users with different levels of digital experience.

6. **Improve field testing**

   * Test the application with real agricultural photographs.
   * Evaluate usability in realistic farming environments.
   * Collect feedback from potential users.
   * Use field observations to guide future model and interface improvements.

The future objective is to develop fellah AI from a prototype into a more robust and field-oriented agricultural assistance tool while maintaining its focus on accessibility and practical use.

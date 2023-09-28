# Multimodal Deep Learning Model:

Multimodal Deep Learning constitutes a specialized domain within machine learning. Its primary objective is to teach artificial intelligence models how to analyze and establish connections among various forms of data, commonly referred to as "modalities". These modalities typically encompass images, videos, audio, and text. By integrating these diverse modalities, a deep learning model becomes capable of comprehending its surroundings in a more comprehensive manner, as specific cues or information may be present in one modality but not in others. 

### Example:

Consider the challenge of emotion recognition. It extends beyond merely analyzing a human face, which falls under the visual modality. The tonal quality and pitch of an individual's voice, categorized under the audio modality, carry substantial information pertaining to their emotional condition. This information might not always be apparent through their facial expressions, even though these two modalities are frequently synchronized in conveying emotions. 

Within the realm of clinical practice, there exist various modalities for acquiring data. These modalities encompass imaging techniques such as computed tomography (CT) scans and X-ray images. Additionally, non-image modalities encompass data sources such as electroencephalogram (EEG) data. Furthermore, this list can be expanded to include sensor-derived data, such as thermal data or information obtained from eye-tracking devices.

### **The common combination of modalities are as follows:** 

Image + Text 

Image + Audio 

Image + Text + Audio 

Text + Audio 

### **Fusion:**

Fusion involves the process of amalgamating information from two or more modalities to accomplish predictive tasks. Effectively combining multiple modalities like video, speech, and text poses a formidable challenge due to the diverse nature of multimodal data.  

The fusion of heterogeneous information constitutes the crux of multimodal research but is accompanied by a plethora of difficulties. Practical challenges encompass addressing issues like disparate data formats, varying data lengths, and non-synchronized data streams. Theoretical challenges revolve around determining the most optimal fusion technique, which ranges from straightforward methods like concatenation or weighted summation to more sophisticated approaches like employing attention mechanisms such as transformer networks or attention-based recurrent neural networks (RNNs). 

Additionally, a critical decision to be made is whether to opt for early or late fusion. Early fusion integrates features right after their extraction, employing some of the fusion methods mentioned earlier. Conversely, in late fusion, integration occurs only after each unimodal network produces predictions, be it for classification or regression tasks. Late fusion typically involves techniques like voting schemes, weighted averages, or other methods. Hybrid fusion techniques have also emerged, which blend outputs from early fusion and unimodal predictors, offering a diverse approach to multimodal data integration.


## Early Fusion:

In the early fusion approach with deep neural networks, where you combine both structured data and image data at the beginning of the network architecture, the form of the data that is sent into the final classification layer depends on how you choose to merge or concatenate the representations from the two modalities. Typically, you'll have two separate branches for each modality, and the form of data sent to the final classification layer can be one of the following:

**Concatenation:** In this case, you concatenate or stack the representations from the two modalities along a new axis. For example, if the structured data representation is a vector of size N and the image data representation is a vector of size M, the concatenated representation sent to the final classification layer would be a vector of size N + M.

[Structured Data Representation] + [Image Data Representation]


**Merging:** Instead of concatenation, you can merge the representations using a specific operation like element-wise addition or element-wise multiplication. This operation combines the information from both modalities in a way that you believe is beneficial for the task.

Merged Representation = f([Structured Data Representation], [Image Data Representation])


**Multi-Modal Fusion Layer:** You can add a specific layer that learns to fuse the representations from both modalities. This can be a fully connected layer or a custom layer designed for fusion. The output of this fusion layer would then be sent to the final classification layer.

[Structured Data Representation]----|

                                |------   [Fusion Layer] --- [Final Classification Layer]

[Image Data Representation]---------| 


**Normalization and Transformation:** Before sending the fused representation to the final classification layer, you may apply normalization techniques or further transformations to ensure that the data is in a suitable format for the classification task. This might involve applying activation functions or additional layers.



 ### **Working procedure of Multimodal Deep Learning Model:**

Unimodal Encoders: These neural networks are responsible for processing and encoding data from individual modalities. Typically, there is one encoder for each input modality, ensuring that data from different sources is handled separately. 

Fusion Network: Following the unimodal encoding phase, the extracted information from each modality needs to be effectively merged or fused together. Various fusion techniques exist, ranging from straightforward methods like concatenation to more advanced approaches such as attention mechanisms. This fusion module is crucial in harmonizing the diverse modalities. 

Classifier: Once fusion occurs, the fused encoded information is passed on to a final "decision" network. This classifier is trained for the specific end task, such as classification or prediction, making use of the integrated data from the previous steps. 

In summary, the multimodal architectures are structured into these three essential modules: the unimodal encoders, responsible for encoding individual modalities; the fusion module, which combines the encoded features from different modalities; and the classification module, which makes predictions based on the fused data. 


# How exactly a Multimodal Deep Learning model works:


Building a multimodal deep learning model that combines image data and structured data for classification involves several steps. In this example, we'll use a common approach called a fusion model, where we process both types of data separately and then merge their representations to make a final classification. Here's a clear working procedure for creating such a model:

## Data Collection and Preprocessing:

Collect your image data and structured data. Ensure that they are properly labeled for the classification task.
Preprocess the image data by resizing, normalizing pixel values, and augmenting data if needed (e.g., rotation, flipping).
Preprocess structured data by handling missing values, encoding categorical variables (e.g., one-hot encoding), and scaling numerical features.

## Data Splitting:

Split your dataset into training, validation, and test sets. Common splits are 70-80% for training, 10-15% for validation, and 10-15% for testing.

## Model Architecture:

Create separate branches for handling image data and structured data.
For the image branch, use a convolutional neural network (CNN) to extract features from images. You can use popular architectures like VGG, ResNet, or custom architectures.
For the structured data branch, design a neural network or deep learning model that can handle tabular data. This can be a feedforward neural network (FNN) or even a recurrent neural network (RNN) if there is temporal data involved.

## Feature Extraction and Fusion:

Train the image branch and structured data branch separately using their respective training datasets.
After training, extract features from the last layers of both branches. These features will represent the high-level information learned from each data type.
Combine these features, for example, by concatenating them or using a fusion layer with fully connected neural networks.

## Classification Layer:

Add a classification layer on top of the fused features. This can be a fully connected layer with softmax activation for a classification task.
Define the number of output units in the classification layer equal to the number of classes in your classification problem.

## Model Training:

Train the entire multimodal model using the combined training dataset. Use a suitable loss function (e.g., categorical cross-entropy) and an optimizer (e.g., Adam).
Monitor the validation set for overfitting and adjust hyperparameters or use techniques like dropout or regularization if needed.

## Model Evaluation:

Evaluate the model on the test set using metrics like accuracy, precision, recall, F1-score, and confusion matrix.
Analyze the results and fine-tune the model as necessary.
# Disease-Prediction-with-NLP
This is an ongoing academic project turned personal project to create an interactive system that uses NLP patient symptom descriptions to predict most likely disease with associated likelihood. It leverages information from a Disease Oncology, actual free-text patient symptom descriptions and advanced BERT models to do so. I've added a breakdown of the key technologies and libraries used and a summary of the project so far.

## Key Technologies and Libraries
1. Base Libararies:
  **Pandas:** For data manipulation and analysis,
  **TQDM:** For progress bars during data processing,
  **NumPy:** For numerical operations,
  **Scikit-Learn (sklearn):** For implementing classical machine learning models and evaluation metrics,
  **NLTK:** For natural language preprocessing, including tokenization and lowercasing,
  **TensorFlow:** For deep learning model training and implementation,
  **Keras:** For building and fine-tuning deep learning models,

2. Machine Learning Frameworks:
  **Torch (PyTorch):** For deep learning model implementation and training,
  **Transformers (by Hugging Face):** For leveraging pretrained models like BERT and embeddings,
  **Torchtext:** For efficient text processing in NLP tasks,
  **Datasets:** To access and handle NLP datasets effectively.

4. Natural Language Processing:
  **NLTK:** For tokenization, lowercasing, and other text preprocessing tasks,
  **SentencePiece:** For advanced tokenization, particularly useful for working with subword units in pretrained models,
  **SPARQLWrapper:** For accessing and querying disease ontology datasets,
  **Gensim:** For working with pretrained word embeddings like BioWordVec and vectorizing text data,
  **NLPAug:** For data augmentation techniques like synonym replacement, random insertion, and deletion to improve model robustness.

4. Embeddings and Pretrained Models:
  **BioWordVec:** Domain-specific pretrained word embeddings trained on biomedical corpora (Pub-Med and MeSH)
  **Hugging Face’s Transformers:** For contextual embeddings using models like BERT, fine-tuned for biomedical NLP tasks.

## Summary
### Objective
The goal of the project is to predict diseases based on patient symptom descriptions using advanced NLP techniques and leveraging state-of-the-art embeddings and machine learning models.

### Approach
The data was collected from a variety of sources and combined into a master dataset, after which the text was cleaned and preprocessed with tools like NLTK for tokenization and lowercasing, while Pandas handled the heavy lifting for data organization and analysis. For turning text into numerical representations, I relied on pretrained embeddings like BioWordVec, supported by Gensim and SentencePiece for advanced tokenization.

The core of the project’s machine learning work revolved around both classical and deep learning approaches. Traditional models built using scikit-learn and TF-IDF embeddings served as useful benchmarks, while TensorFlow, Keras, and PyTorch powered the deep learning models. Hugging Face’s Transformers library, specifically BERT brought in pretrained contextual embeddings that were fine-tuned to better capture the nuances of biomedical text. To make the models more robust and capable of handling variability in symptom descriptions, I used data augmentation techniques through NLPAug. It works by introducing methods like synonym replacement, random insertion, and deletion, thereby enriching the dataset and leading to improved performance.

As an added feature, I tried to integrate a disease ontology containing structured disease knowledge from ontologies to improve the model's understanding of the relationships between symptoms and diseases. The results showed that pretrained embeddings and fine-tuned deep learning models significantly outperformed the traditional benchmarks, achieving high accuracy, precision, and recall, highlighting how a combination of machine learning, domain-specific knowledge, and creative data strategies can make a big impact in biomedical NLP.

There’s potential to expand the project further by integrating zero-shot learning, enabling the system to predict diseases it hasn’t seen in the training data. At it's current state, I have benchmarked the model against established tools like WebMD and ChatGPT to see how it stacks up against industry standards and it's predictions are on par with ChatGPTS Medical Diagnosis assistant. Overall, this work showcases the power of modern NLP in solving real-world healthcare challenges.

### Findings and Results
While over 90% of the models predictions matched ChatGpt's, some of them, specifically the less detailed patient symptom descriptions got predictions for diseases on the extreme end of the symptoms. E.g A patient who has reported a headache and fever with no other information provided could get a prediction for an 85% chance of hypertension. While these could be symptoms of hypertension, they are more likely to be other symptoms and hypertension would be an extreme case of those symptoms.

### Future Work
1. The original dataset will be expanded to include more diverse symptom descriptions and incorporate symptom severity
2. Zero-shot learning will be incorporated to predict diseases not present in the training data
3. Model will be integrated with ChatGPT to enhance interactivity in the user experience

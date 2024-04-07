# LAB3

 
## Objective
The objective of this project is to build deep neural network architectures for Natural Language Processing (NLP) tasks using PyTorch. Specifically, the project consists of two parts:

## Classification Task: 
Scraping text data from Arabic websites, preprocessing the data, and training classification models using RNN, Bidirectional RNN, GRU, and LSTM architectures.
Text Generation with Transformer: Fine-tuning the GPT2 pre-trained model for text generation on a customized dataset.
### Part 1: Classification Task
#### 1. Data Collection
Text data is collected from Arabic websites using scraping libraries like Scrapy or BeautifulSoup.
Each text is assigned a score representing its relevance, ranging from 0 to 10.
#### 2. Preprocessing
Establish an NLP pipeline including tokenization, stemming, lemmatization, stop words removal, and discretization.
#### 3. Model Training
Train models using RNN, Bidirectional RNN, GRU, and LSTM architectures.
Hyperparameters tuning to achieve optimal performance.
#### 4. Evaluation
Evaluate the models using standard metrics and additional metrics like BLEU score.
### Part 2: Transformer (Text Generation)
#### 1. GPT2 Fine-Tuning
Install pytorch-transformers and load the GPT2 pre-trained model.
Fine-tune the model on a customized dataset for text generation.
#### 2. Text Generation
Generate new paragraphs based on given sentences using the fine-tuned GPT2 model.
## Files Included
data_scraper.py: Contains code for scraping data from Arabic websites.
preprocessing.py: Implements the NLP preprocessing pipeline.
classification_models.py: Defines RNN, Bidirectional RNN, GRU, and LSTM models for the classification task.
evaluation_metrics.py: Evaluates models using standard metrics and BLEU score.
transformer_finetuning.py: Fine-tunes the GPT2 model for text generation.
text_generation.py: Generates paragraphs based on input sentences using the fine-tuned GPT2 model.
Usage
Clone the repository.
Install required libraries using pip install -r requirements.txt.
Run the scripts sequentially or as needed, adjusting parameters as necessary.
## Results
Results and performance metrics are stored in respective output files or printed to the console.
Detailed analysis and interpretation of results can be found in the respective sections of the code or in accompanying documentation

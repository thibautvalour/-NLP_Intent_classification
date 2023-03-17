# NLP Intent classification

This is a simple NLP project to classify the intent of a sentence in a dialogue. The purpose of this project is to showcase the implementation of two models for intent classification using natural language processing techniques.

## Dataset

The dataset used in this project is silicone from Huggingface datasets : https://huggingface.co/datasets/silicone

## Models

Two models were implemented in this project:

- Baseline model: A model using Bert with a linear layer on top of it. Accuracy on validation set: 60%.
- BertBiLSTM model: LSTM model on top of the Bert in order to take into account the context of the sentences. Accuracy on validation set: 85%.

<img src="https://github.com/thibautvalour/NLP_Intent_classification/blob/main/Architecture.png" width="500">

More details about the models and their results can be found in `report.pdf`

## Running the code

To run the code in this project, follow these steps:

- Clone the repository.
- Install the required dependencies.
- Run `baseline.ipynb` to view the implementation of the baseline model, this notebook is more detailed.
- Run `main.ipynb` to train and validate the BertBiLSTM model.

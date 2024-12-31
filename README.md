# Document Classification Solution

## Description
This repository contains a document classification solution aimed at classifying scientific papers into three parent categories and eleven subcategories. I utilized the WOS data (5736 samples) provided by [[Kowsari et al. (2017)](http://arxiv.org/pdf/1709.08267v2)].

## Motivation
### Potential Solutions:
1. **Tree-based Models**:
   - Obtain embeddings of abstracts and train a tree-based model (e.g., Random Forest) to predict the category and subcategory in hierarchical order.

2. **Fine-tuning BERT**:
   - Fine-tune a BERT-based model with a classifier layer on top to predict the category and subcategory, potentially in hierarchical steps (parent category first, followed by subcategory).

3. **LLM-based Approaches**:
   - Use large language models (LLMs) in zero-shot or few-shot settings to predict the category and subcategory.

### Motivation for Choosing LLM-based Approach:
- Fine-tuning BERT requires significant computational resources (e.g., GPUs), while LLM-based approaches do not require complex training processes.
- Tree-based models could work but are better suited for simpler classification problems and may face extrapolation issues when encountering new document topics not present in the training data.
- LLM-based models benefit from descriptive inputs for each class, improving accuracy without specialized SME (Subject Matter Expert) knowledge.
- To optimize class descriptions, I leveraged LLMs to analyze and generate optimal descriptions based on examples for each class.


## Goal of the Experiment
This experiment involves two main steps:

1. **Generating Descriptions**: Using a small subset of labeled data (5 samples for each subcategory), descriptions are generated for each subcategory and its parent category based on the questions generated by the llm to find the simmiliarity and differneces between the classes. These descriptions then generated based on the questions and answers based on samples with the true lables by llm and aim to capture the unique characteristics of each class.

2. **Classification Using Descriptions**: The generated descriptions are passed to another LLM to classify the abstracts. The process involves predicting the parent category first and then classifying the abstract into one of the subcategories associated with the predicted parent category.

### Objective
The primary goal is to eliminate human intervention in defining class descriptions, which can be effort-intensive. This aligns with the approach outlined by the Bardeen team in their referenced work [[citation](http://arxiv.org/pdf/2310.06111)]. Given the 2-3 hour constraint for this experiment, this solution provides a potential alternative.


## Evaluation Results
We evaluated the solution using:
- 5 samples per subcategory for generating descriptions.
- 275 test samples for classification.

Below is the classification report for parent and child category predictions:

| Metric (weighted ave)       | Parent Category | Child Category |
|---------------|-----------------|----------------|
| Precision     | 0.87         | 0.75       |
| Recall        | 0.85         | 0.65        |
| F1-Score      | 0.86         | 0.63        |
| Accuracy      | 0.86         | 0.64        |
| Number of Samples | 275         | 275            |

These results represent that this moodel has achieved some improvments in compare to the zero shot approach by 6% in accuracy. (although that the test set may not be same as the zero shot experiment mentioned in the paper)

## Limitations
1. **Time Constraints**:
   - There is room for better prompting to extract more meaningful information for description generation.

2. **Budget (LLM Cost)**:
   - OpenAI API was used as the LLM model. Due to API costs, the code was not run on the entire dataset.
   - The LLaMA 3.2 model (3 billion parameters) was also tested using the Ollama framework. However, its performance did not match GPT-4o, which was expected given the parameter size difference.

## Next Steps
1. Develop better prompts to capture detailed and accurate information for each class.
2. Implement more robust error management throughout the code. This area was deprioritized due to time constraints.
3. Better modularity within IterativeDescriptionLearner class.
4. Seperate the training and inference pipeline

---

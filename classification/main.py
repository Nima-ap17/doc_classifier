"""Main script to run the classification pipeline"""
from models.iterative_learner import IterativeDescriptionLearner
from models.document_classifier import DocumentClassifier
from utils.data_loader import load_data, load_config
from models.schema import output_class_mapping
from utils.evaluation import calculate_metrics, save_evaluation_results
from sklearn.model_selection import train_test_split
from langchain_openai import ChatOpenAI
import pandas as pd
from pathlib import Path
import json
import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger()

def main():
    config = load_config()

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Load and prepare data
    logger.info('Loading and preparing data')
    data = load_data(config['data']['x_path'], config['data']['y1_path'], config['data']['y2_path'], 
                    config['category_mappings'], config['subcategory_mappings'])
    
    # Sample examples from each class as it would be costly to train and test on the whole entire dataset
    data = data.groupby('child_label').apply(
        lambda x: x.sample(min(len(x), 30))
    )

    # Split data into train and test sets with stratification
    train_data, test_data = train_test_split(
        data,
        test_size=(len(data) - 55)/len(data), # 55 samples for training, 5 per each class
        stratify=data['child_label'], 
        # random_state=42 
    )
    
    # Initialize the llm
    llm = ChatOpenAI(
        model=config['model']['name'],
        temperature=config['model']['temperature']
    )
    # Train the model
    logger.info('Training the description learner')
    learner = IterativeDescriptionLearner(llm=llm, output_class_mapping=output_class_mapping)
    parent_desc, child_desc = learner.learn_descriptions(
        data=train_data, text_column='text', parent_label_col='parent_label', child_label_col='child_label')
    
    # Initialize classifier with learned descriptions
    logger.info('Initializing the document classifier')
    classifier = DocumentClassifier(llm=llm, parent_desc=parent_desc, child_desc=child_desc)
    
    # Make predictions
    logger.info('Making predictions')
    test_data[['prediction_parent', 'prediction_child']] = test_data['text'].apply(
        classifier.classify_abstract
    ).apply(pd.Series)


    # Evaluate results
    logger.info('Evaluating results')
    metrics_l1 = calculate_metrics(test_data,y_true_col='parent_label', y_pred_col='prediction_parent')
    metrics_l2 = calculate_metrics(test_data,y_true_col='child_label', y_pred_col='prediction_child')
    
    # Save results
    test_data.to_csv(output_dir / 'predictions.csv', index=False)
    save_evaluation_results(metrics_l1, output_dir / 'evaluation_l1.txt')
    save_evaluation_results(metrics_l2, output_dir / 'evaluation_l2.txt')

if __name__ == '__main__':
    main()

"""This module contains the IterativeDescriptionLearner class, 
which is responsible for learning descriptions for parent and child categories based on the provided data."""
import pandas as pd
from typing import Dict, Tuple
from models.schema import (Description, Questions, Answers)

class IterativeDescriptionLearner:
    def __init__(self, llm, output_class_mapping):
        """
        Initialize the IterativeDescriptionLearner with the provided language model.
        param llm: The language model to use for learning descriptions
        """
        self.llm = llm
        self.class_descriptions: Dict = {}
        self.output_class_mapping = output_class_mapping

    def learn_descriptions(
        self, 
        data: pd.DataFrame, 
        text_column: str, 
        parent_label_col: str,
        child_label_col: str,
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
        """
        Learn descriptions for the parent and child categories based on the provided data.
        param data: The input data containing the text and labels
        param text_column: The name of the column containing the text data
        param parent_label_col: The name of the column containing the parent labels
        param child_label_col: The name of the column containing the child labels
        return: A tuple containing the descriptions for the parent and child categories
        """
        parent_labels = list(data[parent_label_col].unique())
        questions = self._generate_initial_question(labels=parent_labels)
        
        # Sample examples from each class as it would be costly to train and test on the whole entire dataset
        parent_data = data.groupby(parent_label_col).apply(
            lambda x: x.sample(min(len(x), 10))
        )
        answers = []
        for _, row in parent_data.iterrows():
            answers.append(self._answer_initial_question(
                row[text_column], row[parent_label_col], parent_labels, questions
            ))

        parent_categories_desc = self._generate_description(
            data[parent_label_col].unique(), 
            questions, answers, Description
        )
        
        child_descriptions = {}
        
        for cat in parent_labels:
            parent_df = data[data[parent_label_col] == cat]
            
            labels = list(data[data[parent_label_col] == cat][child_label_col].unique())
            questions = self._generate_initial_question(labels=labels)
            
            answers = []
            for _, row in parent_df.iterrows():
                answers.append(self._answer_initial_question(
                    row[text_column], row[child_label_col], labels, questions
                ))
            
            structured_class = self.output_class_mapping[cat]
            descriptions = self._generate_description(
                parent_df[child_label_col].unique(), 
                questions, answers, structured_class
            )
            child_descriptions[cat] = descriptions
        
        return parent_categories_desc, child_descriptions

    def _generate_initial_question(self, labels: list) -> Questions:
        """
        Generate initial questions to identify potential ambiguities and similarities between categories.
        param labels: The list of categories to analyze
        return: The generated questions
        """
        prompt = f'''
            You are a classification expert tasked with creating questions to identify potential ambiguities and similarities between specific categories. 
            These questions should help a classifier agent better distinguish between categories.

            ### Input Details:
            1. **Categories**: A list of categories to analyze. For example: '{labels}'.
            2. **Purpose**: The questions should focus on possible areas of overlap, ambiguity, or similarities between the categories, targeting key concepts, methodologies, or themes.

            ### Your Task:
            Generate **four questions** in total, considering all categories collectively. These questions should:
            1. Highlight potential overlaps or ambiguous areas among the categories.
            2. Explore unique features or methodologies that can differentiate one category from the others.
            3. Be designed to help a classifier agent better understand and distinguish between the categories.

            ### Output Format:
            ```
            1. Question 1: <Insert question targeting ambiguity or similarity across categories>
            2. Question 2: <Insert question targeting ambiguity or similarity across categories>
            3. Question 3: <Insert question targeting ambiguity or similarity across categories>
            4. Question 4: <Insert question targeting ambiguity or similarity across categories>
            ```

            Generate four such questions based on the provided categories to assist the classifier agent.
            '''
        
        structured_llm = self.llm.with_structured_output(Questions)
        return structured_llm.invoke(prompt)

    def _answer_initial_question(
        self, 
        document: str, 
        label: str, 
        labels: list, 
        questions: str
    ) -> Answers:
        """
        Answer the initial questions to identify potential ambiguities and similarities between categories.
        param document: The document text to analyze
        param label: The label of the document
        param labels: The list of categories to analyze
        param questions: The questions to answer
        return: The answers to the questions
        """
        prompt = f'''
            You are an expert in scientific literature classification. 
            Your task is to answer questions that identify potential ambiguities and similarities between specific categories. 
            These questions should help a classifier agent better distinguish between categories.

            ### Input Details:
            1. **Categories**: A list of categories to analyze. For example: '{labels}'.
            2. **Questions**: A set of questions designed to explore overlaps, ambiguities, or unique features among the categories. '{questions}'

            ### Your Task:
            Based on the provided questions and example for corresponding category below, answer each question to:
            1. Clarify potential overlaps or ambiguous areas among the categories.
            2. Highlight unique features or methodologies that can differentiate one category from the others.
            3. Provide insights that will help a classifier agent better understand and distinguish between the categories.

            example document:
            - category: {label}
            - content: {document}

            Answer the questions based on the provided categories and example to assist the classifier agent.
            '''
        
        structured_llm = self.llm.with_structured_output(Answers)
        return structured_llm.invoke(prompt)

    def _generate_description(
        self, 
        labels: str, 
        questions: str, 
        answers: str, 
        output_class
    ) -> Dict[str, str]:
        """
        Generate descriptions for the provided labels based on the questions and answers.
        param labels: The list of labels to generate descriptions for
        param questions: The questions used to generate the descriptions
        param answers: The answers to the questions
        param output_class: The class of the structured output
        return: The descriptions for the provided labels
        """
        prompt = f'''
            You are an expert in scientific literature classification. 
            Your task is to update the generalized descriptions of classes for classifying scientific paper abstracts based on the answers to the questions provided. 
            The descriptions should be:

            - Generalized: Avoid overly detailed to ensure the description is broadly applicable.
            - Distinguishable: Highlight the key characteristics that differentiate each class from the others, focusing on unique themes, concepts, or methodologies.
            - Purpose-Oriented: Designed to guide another language model in correctly categorizing new abstracts into these classes.

            Here are all the classes:
            {labels}
            
            Now I want you to write down the description for each classes based on the answers and questions provided below. the final description should consider all the three items mentioned above:
            questions:
            {questions}

            answers:
            {answers}

            now write down the descriptions for each class in {labels} considering the points mentioned above.
            '''
        
        structured_llm = self.llm.with_structured_output(output_class)
        return structured_llm.invoke(prompt)

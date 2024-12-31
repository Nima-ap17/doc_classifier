"""This module contains functions for classifying documents into categories and subcategories."""
from typing import Tuple
from langchain_core.prompts import ChatPromptTemplate
from models.schema import DocClass

class DocumentClassifier:
    def __init__(self, llm, parent_desc, child_desc):
        """
        Initialize the DocumentClassifier with the learned descriptions.
        param llm: The language model to use for classification
        param parent_desc: Description of the parent categories
        param child_desc: Description of the child categories
        """
        self.llm = llm
        self.parent_desc = parent_desc
        self.child_desc = child_desc
        
        system_prompt = '''You are an assistant that classifies document abstracts. 
            Based on the provided abstract, classify it into the most appropriate category.
        
            categoreies and their descriptions:
            {category_desc}

            Abstract:
            {abstract}

            - Choose the closes category to the abstract from the list of categories.
            - Make sure that the selected category is in the provided categories list above.
        '''

        main_category_prompt = ChatPromptTemplate.from_template(system_prompt)
        structured_llm = self.llm.with_structured_output(DocClass)
        self.chain = main_category_prompt | structured_llm

    def classify_abstract(self, abstract: str) -> Tuple[str, str]:
        """
        Classify the provided abstract into the most appropriate category and subcategory.
        param abstract: The abstract to classify
        return: A tuple containing the main category and subcategory
        """
        try: 
            response_1 = self.chain.invoke({'abstract': abstract, 'category_desc': self.parent_desc})
            response_1_category = response_1.category.lower()
        except:
            return 'No category description found', 'No subcategory description found'
        
        try: 
            subcategory_description = self.child_desc[response_1_category]
            response_2 = self.chain.invoke({'abstract': abstract, 'category_desc': subcategory_description})
            return response_1.category, response_2.category
        
        except KeyError:
            return response_1.category, 'No subcategory description found'
        
        

    
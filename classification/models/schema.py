"""This module contains the Pydantic models for the classification functions."""
from pydantic import BaseModel

class Description(BaseModel):
    biochemistry: str
    electrical_engineering: str
    psychology: str

class Biochemistry(BaseModel):
    molecular_biology: str
    immunology: str
    polymerase_chain_reaction: str
    northern_blotting: str

class ElectricalEngineering(BaseModel):
    electricity: str
    digital_control: str
    operational_amplifier: str

class Psychology(BaseModel):
    social_cognition: str
    child_abuse: str
    attention: str
    depression: str

class Questions(BaseModel):
    Question_1: str
    Question_2: str
    Question_3: str
    Question_4: str

class Answers(BaseModel):
    Answer_for_question_1: str
    Answer_for_question_2: str
    Answer_for_question_3: str
    Answer_for_question_4: str

class DocClass(BaseModel):
    category: str

output_class_mapping = {
    'biochemistry': Biochemistry,
    'electrical_engineering': ElectricalEngineering,
    'psychology': Psychology
}

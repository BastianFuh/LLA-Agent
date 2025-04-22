# Language Learning Assistant Agent Platform (LLA-Agent)

This project aims to create a comprehensive learning assistant by leveraging the power of AI to generate and evaluate questions similar to those seen in conventional learning literature or language exams.

# Design

The system is based on function-calling agents, which iteratively build the questions.  Information is not manually extracted from the LLM output but is given by the model by creating specific function calls. This makes the system more robust and allows errors to be corrected in each step. The relevant question data is collected across multiple steps and handed to the GUI for display.

The evaluation for questions for which the answer can be derived from the generation is done by storing the correct answer. For more complex and open-ended questions, a chatbot is used. It will be given the entire context of the question and the user's response. After it has given its evaluation, further clarification questions about the problem can be asked.

# Current Features

Current question types include:

-	Multiple Choice
-	Fill in the blank
-	Translation with Chabot-based Evaluation
-	Reading Comprehension with Chabot-based Evaluation

Additionally, these questions support optional audio output:

-	Translation
-	Reading Comprehension

Currently supports models from OpenAI and DeepSeek.

# Currently planned features

-	Listening Comprehension. Similar to Reading Comprehension, but with a different style of text. For example, the focus might be on conversations. 
-	Audio input


# How to Use

## Setup

Todo

## Usage Guide

Language, language proficiency (i.e., CEFR A1-C2, JLPT N5-N1), and question difficulty are defined in the sidebar. These values will be used in the prompts to generate the question or text. 

Via an “Additional Information” section, further instructions can be given to the model. This can be used to guide the generation process to specific topics or control the style of a question.

Regarding language proficiency. It seems that the model's understanding is currently quite limited and inconsistent. It might sometimes use a bit more complex phrasing than expected at the specified level.

Regarding difficulty, the effect of this varies quite a bit, and sometimes, it might not seem to have any effect. However, better internal prompting might mitigate this in the future. Before that, the “Additional Information” section could be used to ask for more complex or simple questions by giving more precise instructions.

# Limitations

It’s important to remember that AI is far from perfect, so there is no 100% guarantee that everything it says will be correct. Therefore, it is crucial to double-check important facts. 
There is also a chance that a question might be malformed or just sound 


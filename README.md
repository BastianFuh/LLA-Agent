# Language Learning Assistant Agent Platform (LLA-Agent)

LLA-Agent is an AI-powered platform designed to assist language learners by automatically generating and evaluating questions akin to those found in conventional language learning resources and standardized exams.


## Overview

The platform utilizes function-calling agents to iteratively construct questions. Instead of manually parsing LLM (Large Language Model) output, the system leverages structured function calls, which enhances robustness and allows for correction at each stage of question generation. The relevant question data is accumulated across multiple steps and presented via the GUI.

Evaluation is handled in two ways:

- **Objective questions**: The correct answer is stored and used for automated validation.
- **Subjective/open-ended questions**: A chatbot evaluates responses based on the full context, offering follow-up clarification if needed.


## Current Features

### Supported question types:

- Multiple Choice  
- Fill-in-the-Blank  
- Translation (with chatbot-based evaluation)  
- Reading Comprehension (with chatbot-based evaluation)
- Listening Comprehension 

### Additional capabilities:

- Optional audio input and output for:
  - Translation
  - Reading Comprehension


### Model support:

- **LLMs**
  - **OpenAI** (API)
  - **DeepSeek** (API)
  - **Ollama**, Function calling models (local)
    > ⚠️ Initial testing indicates that smaller models may struggle to follow instructions reliably. Further evaluation with newer or larger models is recommended.

- **Text to Speech**
  - **Kokoro** (Local)
    - Currently only uses Japanese voices
  - **Elevenlabs** (API)
    - Voice configuration is not yet available
    - Offers a limited amount of free credits
    - Subscription-based pricing:
      - Usage-based billing is unlocked at higher subscription tiers.
      - Relatively expensive
  - **Fish Audio** (Local, **Recommended**)
    - Local use via [Fish Speech](https://github.com/fishaudio/fish-speech)
    - Performs well even on modest hardware (e.g., tested on *RTX 4060 Ti*)
    - Supports custom voices via simple file drop:
      - Add an `.mp3` or `.wav` file to the `voice_reference` folder
      - Custom voices showed better quality in testing
    - Potential future support for API-based voices:
      - Current API pricing appears reasonable

## Currently planned features

- **Listening Comprehension**: Focused on conversational texts



## Getting Started

### Setup

> *To be completed.*

## Usage Guide

- **Language & Proficiency**: Select the language and proficiency level (e.g., CEFR A1–C2, JLPT N5–N1) via the sidebar. These settings influence prompt generation.
  
- **Question Difficulty**: Adjustable via the interface; however, effects may vary due to limitations in LLM comprehension. Fine-tuning can be achieved using the **“Additional Information”** field.

- **Additional Instructions**: Customize prompts to guide content focus or question style more precisely.

**Notes**:

- Language proficiency understanding by the model may be inconsistent.
- Difficulty adjustments are not always reflected as expected. Use the “Additional Information” field to enforce complexity levels manually.


## Limitations

While AI provides valuable assistance, it is not infallible. Users should:

- Double-check factual accuracy, especially for critical information
- Be aware that generated questions may occasionally be malformed or awkward


from frontend.tabs.messages.messages import Messages


class ENMessages(Messages):
    """English text strings for the application."""

    def button_generate_question(self) -> str:
        return "Generate Question"

    def question_placeholder(self) -> str:
        return "Your question will be generated here"

    def placeholder_answer_field(self) -> str:
        return "Type your answer..."

    def button_validate_answer(self) -> str:
        return "Validate Answer"

    def placeholder_chatbot(self) -> str:
        return "<center><strong>Your Personal Language Learning Assistant</strong><br>Ask Me Anything</center>"

    def placeholder_chatbot_evaluation(self) -> str:
        return "<center><strong>The answers will be evaluated here</strong><br>You can also ask Me Anything</center>"

    def placeholder_chatbot_input(self) -> str:
        return "Type a message..."

    def label_question(self) -> str:
        return "Question"

    def info_generate_question_first(self) -> str:
        return "Please generate a question first."

    def info_correct_answer(self) -> str:
        return "Correct Answer"

    def info_incorrect_answer(self) -> str:
        return "Incorrect Answer"

    def label_topic(self) -> str:
        return "Topic"

    def placeholder_comprehension_topic(self) -> str:
        return "The topic of the text will be generated here"

    def placeholder_comprehension_text(self) -> str:
        return "The text will be generated here"

    def label_speaker_1(self) -> str:
        return "Speaker 1"

    def label_speaker_2(self) -> str:
        return "Speaker 2"

    def placeholder_speaker_1_text(self) -> str:
        return "The name of the first speaker"

    def placeholder_speaker_2_text(self) -> str:
        return "The name of the second speaker"

    def label_listening_comprehension_switch(self):
        return "Listening Comprehension"

    def button_show_text(self) -> str:
        return "Show Text"

    def label_answer(self):
        return "Answer"

    def label_answer_numbered(self, number):
        return f"Answer {number}"

    def info_select_option(self):
        return "Please select an option."

    def label_settings(self):
        return "Settings"

    def label_language(self):
        return "Language"

    def label_language_proficiency(self):
        return "Language Proficiency"

    def label_difficulty(self):
        return "Question Difficulty"

    def label_additional_information(self):
        return "Additional Information for Question Generation"

    def tab_label_options(self):
        return "Options"

    def tab_label_conversation_exercise(self):
        return "Conversation Exercise"

    def tab_label_multiple_choice_question(self):
        return "Multiple Choice Fill-in-the-blank"

    def tab_label_fill_in_blank_question(self):
        return "Constructed Response Fill-in-the-blank"

    def tab_label_translation_question(self):
        return "Translation"

    def tab_label_comprehension_monologue_question(self):
        return "Comprehension - Monologue"

    def tab_label_comprehension_dialogue_question(self):
        return "Comprehension - Dialogue"

    def label_model(self):
        return "Model for the Chatbot"

    def label_embedding_model(self):
        return "Embedding Model"

    def label_search_engine(self):
        return "Search Engine"

    def label_tts_provider(self):
        return "TTS Provider"

class CreativeDesignPrompter:
    """
    Responsible for converting canonicalBoard, MCTS thoughts, and ML scores
    into natural language prompts for the LLM.
    """
    def __init__(self, game):
        self.game = game

    def generate_initial_design_prompt(self, canonical_board_str):
        """
        Generates a prompt for initial design evaluation.
        """
        # Placeholder: In a real scenario, canonical_board_str would be detailed features.
        return f"Evaluate the following smart wearable design represented by its features: {canonical_board_str}. Provide insights on its creativity dimensions (Novelty, Utility, Aesthetics, Technical Feasibility, Divergent Thinking, Integration) and suggest initial improvements."

    def generate_revision_prompt(self, canonical_board_str, creative_scores, previous_llm_output, specific_recommendations=None):
        """
        Generates a prompt for revising a design based on feedback.
        """
        feedback_str = ", ".join([f"{dim}: {score}" for dim, score in creative_scores.items()])
        recommendations_str = ". ".join(specific_recommendations) if specific_recommendations else "None."

        prompt = (
            f"Given the current smart wearable design with features: {canonical_board_str}. "
            f"Previous evaluation indicated: {previous_llm_output}. "
            f"The current creative scores are: {feedback_str}. "
            f"Specific recommendations for improvement: {recommendations_str}. "
            f"Please revise the design concept based on this feedback, focusing on improving the identified weaknesses and incorporating the recommendations."
            f" Generate a new, improved design description."
        )
        return prompt

    def xot_prompt_multi_wrap(self, x, thoughts_list):
        """
        Wraps multiple thoughts into a prompt for LLM.
        """
        # Placeholder: x would be initial design description, thoughts_list are MCTS actions/intermediate steps
        thoughts_str = "; ".join(thoughts_list)
        return f"Based on the initial design concept '{x}' and the following thought process: '{thoughts_str}', generate a detailed smart wearable design description, focusing on maximizing creativity.", "instruction"

    def get_instruction_prompt(self):
        """
        Returns a general instruction prompt for the LLM.
        """
        return "You are an expert in smart wearable design. Provide creative, innovative, and practical design solutions."

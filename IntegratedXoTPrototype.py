import numpy as np
import openai # Assuming openai library is available for LLM interaction
import os

# Import previously defined classes
from CreativeDesignGame import CreativeDesignGame, preprocess_design_data
from CreativeDesignNNet import CreativeDesignNNet
from CreativeDesignPrompter import CreativeDesignPrompter
from CreativeDesignParser import CreativeDesignParser

# Mock OpenAI LLM call for demonstration purposes
# In a real scenario, this would use openai.ChatCompletion.create
def mock_llm_call(prompt_text, instruction_text=""):
    print(f"\n--- MOCK LLM CALL ---\nInstruction: {instruction_text}\nPrompt: {prompt_text[:200]}...\n---")
    # Simulate LLM response based on prompt type
    if "initial improvements" in prompt_text:
        return "LLM Response: Initial design concept is novel but lacks aesthetic appeal. Consider incorporating modern minimalist design principles."
    elif "revise the design concept" in prompt_text:
        return "LLM Response: Revised design now features a sleek, modular build with customizable aesthetic elements, addressing previous feedback."
    else:
        return "LLM Response: General design description based on thoughts."

class SimulatedXoTSolver:
    """
    A simplified simulation of the XoT_Solver to demonstrate integration.
    """
    def __init__(self, initial_design_data, game_class, nnet_class, prompter_class, parser_class, gpt_model=None):
        self.game = game_class(initial_design_data)
        self.nnet = nnet_class(self.game) # Our ML model implementing NeuralNet
        self.prompter = prompter_class(self.game)
        self.parser = parser_class(self.game)
        self.gpt = gpt_model if gpt_model else mock_llm_call # Use mock LLM if no real GPT is provided
        self.current_canonical_board = self.game.canonicalBoard
        self.step = 0
        self.llm_history = []
        self.initial_design_data = initial_design_data # Store initial design data

    def _simulate_mcts_thoughts(self, current_board_str, num_thoughts=3):
        """
        Simulates MCTS generating a sequence of thoughts (actions).
        """
        thoughts = []
        for _ in range(num_thoughts):
            # Ensure the input to nnet.predict is consistent with what preprocess_design_data expects
            # For this simulation, we'll pass a dummy dict that preprocess_design_data can handle
            dummy_design_data = {"concept": f"design_state_{current_board_str[:10]}"}
            pi, _ = self.nnet.predict(preprocess_design_data(dummy_design_data))
            action = np.argmax(pi) # Select action based on predicted policy
            thought_text = self.parser.action_to_thoughs(action, current_board_str)
            thoughts.append(thought_text)
        return thoughts, action # Return last action for simplicity in getNextState call

    def solve(self, design_id="default_design"):
        print(f"\n--- Starting XoT Creative Design Process for {design_id} ---")

        # --- Initial Evaluation Phase ---
        print("\nPhase 1: Initial Design Evaluation")
        initial_board_str = self.game.stringRepresentation(self.current_canonical_board)
        initial_prompt = self.prompter.generate_initial_design_prompt(initial_board_str)
        instruction_prompt = self.prompter.get_instruction_prompt()

        # LLM generates initial insights
        initial_llm_response = self.gpt(initial_prompt, instruction_prompt)
        self.llm_history.append(initial_llm_response)
        print(f"LLM Initial Insights: {initial_llm_response}")

        # --- MCTS-driven Design Exploration and Iteration Phase ---
        print("\nPhase 2: MCTS-driven Design Exploration and Iteration")
        is_design_satisfactory = False
        max_iterations = 3

        while not is_design_satisfactory and self.step < max_iterations and not self.game.getGameEnded(self.current_canonical_board, self.step):
            self.step += 1
            print(f"\nIteration {self.step}: Generating new thoughts and revisions...")

            # Simulate MCTS generating thoughts (actions)
            mcts_thoughts, last_action = self._simulate_mcts_thoughts(self.game.stringRepresentation(self.current_canonical_board))
            print(f"  MCTS Thoughts (Actions): {mcts_thoughts}")

            # Simulate getting next state (design evolution)
            self.current_canonical_board = self.game.getNextState(self.current_canonical_board, last_action)
            current_board_str_for_llm = self.game.stringRepresentation(self.current_canonical_board)

            # Prompter uses thoughts to generate a more detailed prompt for LLM
            # Access initial_design_data from instance attribute
            xot_llm_prompt, xot_llm_instruction = self.prompter.xot_prompt_multi_wrap(self.initial_design_data["concept"], mcts_thoughts)
            current_llm_response = self.gpt(xot_llm_prompt, xot_llm_instruction)
            self.llm_history.append(current_llm_response)
            print(f"  LLM Design Description based on thoughts: {current_llm_response}")

            # Parser evaluates LLM's response using simulated ML scores
            evaluation_result = self.parser.test_output(current_llm_response, current_board_str_for_llm)
            is_design_satisfactory = evaluation_result["is_correct"]
            creative_scores = evaluation_result["creative_scores"]
            insights_report = evaluation_result["key_insights_report"]
            overall_rating = evaluation_result["overall_creative_rating"]

            print(f"  ML-based Evaluation: Overall Rating = {overall_rating}, Satisfactory = {is_design_satisfactory}")
            print(f"  Detailed Scores: {creative_scores}")
            print(f"  Strengths: {insights_report['strengths']}")
            print(f"  Weaknesses: {insights_report['areas_for_improvement']}")
            print(f"  Recommendations: {insights_report['specific_recommendations']}")

            if not is_design_satisfactory and self.step < max_iterations:
                # Generate revision prompt based on ML feedback
                revision_prompt = self.prompter.generate_revision_prompt(
                    current_board_str_for_llm,
                    creative_scores,
                    current_llm_response,
                    insights_report["specific_recommendations"]
                )
                revision_llm_response = self.gpt(revision_prompt, instruction_prompt)
                self.llm_history.append(revision_llm_response)
                print(f"  LLM Revision Suggestion: {revision_llm_response}")

        # --- Final Output Phase ---
        print("\n--- Final Design Evaluation and Report ---")
        final_board_str = self.game.stringRepresentation(self.current_canonical_board)
        final_eval = self.parser.test_output(self.llm_history[-1], final_board_str)

        final_output = {
            "design_id": design_id,
            "creative_scores": final_eval["creative_scores"],
            "overall_creative_rating": final_eval["overall_creative_rating"],
            "key_insights_report": final_eval["key_insights_report"],
            "mcts_thought_path": mcts_thoughts, # Simplified, in reality would be the full path
            "raw_llm_outputs_history": self.llm_history
        }

        print(f"\nFinal XoT Output for {design_id}:\n{final_output}")
        return final_output

# --- Main Execution ---
if __name__ == '__main__':
    # Setup for real OpenAI API if key is available
    # if os.getenv("OPENAI_API_KEY"): # Ensure the key is set in your environment variables
    #     openai.api_key = os.getenv("OPENAI_API_KEY")
    #     print("OpenAI API key detected. Using real LLM calls.")
    #     # Define a wrapper for openai.ChatCompletion.create that fits the mock_llm_call signature
    #     def real_llm_call(prompt_text, instruction_text=""):
    #         response = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo", # or "gpt-4"
    #             messages=[
    #                 {"role": "system", "content": instruction_text},
    #                 {"role": "user", "content": prompt_text}
    #             ],
    #             temperature=0.7,
    #             max_tokens=500
    #         )
    #         return response.choices[0].message.content
    #     llm_to_use = real_llm_call
    # else:
    #     print("OpenAI API key not found. Using mock LLM calls.")
    llm_to_use = mock_llm_call # Always use mock for this prototype

    simulated_design_data = {
        "concept": "Smart ring for continuous glucose monitoring with non-invasive sensors",
        "materials": "Titanium alloy, medical-grade silicone",
        "target_user": "Diabetic patients, health-conscious individuals"
    }

    solver = SimulatedXoTSolver(
        initial_design_data=simulated_design_data,
        game_class=CreativeDesignGame,
        nnet_class=CreativeDesignNNet,
        prompter_class=CreativeDesignPrompter,
        parser_class=CreativeDesignParser,
        gpt_model=llm_to_use # Pass the chosen LLM function
    )

    solver.solve(design_id="glucose_monitor_ring")

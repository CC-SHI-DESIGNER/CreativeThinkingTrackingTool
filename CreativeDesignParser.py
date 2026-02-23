import numpy as np

class CreativeDesignParser:
    """
    Responsible for parsing LLM output, extracting creative scores,
    identifying correctness, and generating structured feedback or revision instructions.
    Also converts MCTS actions to human-readable 'thoughts'.
    """
    def __init__(self, game):
        self.game = game

    def test_output(self, llm_output, canonical_board_str):
        """
        Evaluates the LLM's output for correctness/quality based on simulated ML scores.
        In a real scenario, this would involve a sophisticated ML model.
        """
        print(f"Parsing and evaluating LLM output for: {canonical_board_str[:50]}...")
        # Simulate ML-based evaluation by generating random scores
        # In a real setup, an ML model would process llm_output and canonical_board
        np.random.seed(hash(llm_output) % (2**32 - 1)) # Seed for consistent random output for same input

        scores = {
            "Novelty": round(np.random.uniform(50, 95), 2),
            "Utility": round(np.random.uniform(60, 90), 2),
            "Aesthetics": round(np.random.uniform(40, 85), 2),
            "Technical Feasibility": round(np.random.uniform(50, 90), 2),
            "Divergent Thinking": round(np.random.uniform(55, 90), 2),
            "Integration": round(np.random.uniform(65, 95), 2)
        }

        overall_avg = np.mean(list(scores.values()))
        if overall_avg >= 80:
            overall_rating = "高创意"
            is_correct = True # Assuming 'correct' means high creativity
        elif overall_avg >= 60:
            overall_rating = "中创意"
            is_correct = False
        else:
            overall_rating = "低创意"
            is_correct = False

        strengths = f"设计在{max(scores, key=scores.get)}方面表现出色。"
        weaknesses = f"在{min(scores, key=scores.get)}方面有提升空间。"
        recommendations = [
            f"建议专注于提高{min(scores, key=scores.get)}的分数。"
        ]

        # In a real scenario, this would trigger XoT's revision loop if not 'correct'
        print(f"   Simulated Evaluation Result: {'Correct' if is_correct else 'Needs Revision'}")
        return {
            "is_correct": is_correct,
            "creative_scores": scores,
            "overall_creative_rating": overall_rating,
            "key_insights_report": {
                "strengths": strengths,
                "areas_for_improvement": weaknesses,
                "specific_recommendations": recommendations
            }
        }

    def action_to_thoughs(self, action, current_design_description):
        """
        Converts a numerical MCTS action into a human-readable 'thought' (textual description).
        """
        action_map = {
            0: "Explore new material options",
            1: "Integrate advanced sensor technology",
            2: "Refine user interface for simplicity",
            3: "Optimize ergonomic design",
            4: "Reduce manufacturing cost",
            5: "Enhance battery life",
            6: "Improve aesthetic appeal",
            7: "Add new functional module",
            8: "Streamline communication protocols",
            9: "Consider modularity for future upgrades"
        }
        thought = action_map.get(action, f"Apply a generic design modification (Action {action})")
        return f"Thought: {thought} for current design '{current_design_description[:50]}'"

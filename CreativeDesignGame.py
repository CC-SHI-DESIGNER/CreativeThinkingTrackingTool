import numpy as np
import hashlib

def preprocess_design_data(design_data):
    text_embedding_dim = 64
    image_feature_dim = 64
    structured_feature_dim = 32

    feature_parts = []

    text_description = design_data.get("concept", "") + " " + design_data.get("materials", "")
    if text_description:
        text_seed = int(hashlib.sha256(text_description.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
        np.random.seed(text_seed)
        feature_parts.append(np.random.rand(text_embedding_dim))
        print(f"  Simulated text feature extraction from: '{text_description[:50]}...' ")
    else:
        feature_parts.append(np.zeros(text_embedding_dim))

    if "image_features" in design_data and isinstance(design_data["image_features"], np.ndarray):
        img_features = design_data["image_features"][:image_feature_dim]
        feature_parts.append(img_features)
        print(f"  Using provided image features.")
    elif "image_path" in design_data and design_data["image_path"]:
        img_seed = int(hashlib.sha256(design_data["image_path"].encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
        np.random.seed(img_seed)
        feature_parts.append(np.random.rand(image_feature_dim))
        print(f"  Simulated image feature extraction from path: '{design_data['image_path']}'")
    else:
        np.random.seed(42)
        feature_parts.append(np.random.rand(image_feature_dim))
        print("  Simulated default image features (no specific image provided).")

    structured_data = design_data.get("structured_data", {})
    if structured_data:
        structured_features = np.zeros(structured_feature_dim)
        structured_features[0] = structured_data.get('cost_estimate', 0) / 1000
        structured_features[1] = 1 if 'flexible' in design_data.get('materials', '').lower() else 0
        feature_parts.append(structured_features)
        print(f"  Simulated structured data feature extraction.")
    else:
        feature_parts.append(np.zeros(structured_feature_dim))

    canonical_board = np.concatenate(feature_parts)
    print(f"  Final canonicalBoard created with dimension: {len(canonical_board)}")

    return canonical_board

class CreativeDesignGame:
    """
    Defines the 'game' interface for creative wearable design.
    Each 'state' is a design represented by a canonicalBoard.
    """
    def __init__(self, initial_design_data):
        print("Initializing CreativeDesignGame...")
        self.canonicalBoard = preprocess_design_data(initial_design_data)
        self.action_size = 10
        self.board_size = len(self.canonicalBoard)
        print(f"CreativeDesignGame initialized. Board size: {self.board_size}, Action size: {self.action_size}")

    def getActionSize(self):
        return self.action_size

    def getGameEnded(self, canonicalBoard, step):
        max_steps = 5
        if step >= max_steps:
            return 1
        return 0

    def getValidMoves(self, canonicalBoard):
        return np.ones(self.action_size)

    def getNextState(self, canonicalBoard, action):
        new_board = canonicalBoard.copy()
        perturbation_strength = 0.1
        new_board += np.random.randn(self.board_size) * perturbation_strength
        return new_board

    def stringRepresentation(self, canonicalBoard):
        return str(canonicalBoard.tolist())

    def getBoardSize(self):
        return self.board_size

    def getOneTestBoard(self, idx):
        test_design_data = {
            "concept": f"Test design concept {idx}",
            "materials": "randomized materials",
            "image_path": f"./mock_image_{idx}.png",
            "structured_data": {"cost_estimate": np.random.randint(50, 500)}
        }
        return preprocess_design_data(test_design_data)

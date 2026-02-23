import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Placeholder NeuralNet base class if xot_mcts.NeuralNet is not directly importable
class NeuralNet:
    def __init__(self, game):
        pass
    def train(self, examples):
        pass
    def predict(self, board):
        pass
    def save_checkpoint(self, folder, filename):
        pass
    def load_checkpoint(self, folder, filename):
        pass

# Define a simple Multi-Head Neural Network for Creative Design
class MultiHeadCreativeModel(nn.Module):
    def __init__(self, input_dim, action_size):
        super(MultiHeadCreativeModel, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Policy head (predicts probabilities for each creative action)
        self.policy_head = nn.Linear(64, action_size)

        # Value head (predicts overall creative value of the design state)
        self.value_head = nn.Linear(64, 1) # Output a single scalar value

    def forward(self, x):
        features = self.shared_layers(x)
        pi_logits = self.policy_head(features)
        v = torch.tanh(self.value_head(features)) # Tanh to scale value to [-1, 1]
        return pi_logits, v

class CreativeDesignNNet(NeuralNet):
    """
    Implements the NeuralNet interface for Creative Design using a MultiHeadCreativeModel.
    """
    def __init__(self, game, model_checkpoint=None):
        self.game = game
        self.nnet = MultiHeadCreativeModel(game.getBoardSize(), game.getActionSize())
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.001) # Example learning rate
        # For policy, CrossEntropyLoss expects target class indices for classification,
        # or target distribution for KL divergence if combined with log_softmax.
        # If mcts_pis is a distribution, we use a custom loss below.
        self.criterion_pi = nn.CrossEntropyLoss() # Initial placeholder
        self.criterion_v = nn.MSELoss() # For value (creative score)

        if model_checkpoint:
            self.load_checkpoint(model_checkpoint)

        self.nnet.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.nnet.train() # Set to training mode initially

    def predict(self, canonicalBoard):
        """
        Receives canonicalBoard, runs through NN, outputs policy pi and value v.
        """
        self.nnet.eval() # Set to evaluation mode
        board_tensor = torch.tensor(canonicalBoard, dtype=torch.float32).unsqueeze(0)

        # Ensure tensor is on the correct device
        device = next(self.nnet.parameters()).device
        board_tensor = board_tensor.to(device)

        with torch.no_grad():
            pi_logits, v = self.nnet(board_tensor)

        # Convert policy logits to probabilities
        pi = torch.softmax(pi_logits, dim=1).squeeze(0).cpu().numpy()
        v = v.item() # Extract scalar value

        return pi, v

    def train(self, examples):
        """
        Trains the neural network using examples (board, pi, v) from MCTS self-play.
        """
        self.nnet.train() # Set to training mode

        # examples: list of (canonicalBoard, mcts_pi, mcts_v)
        boards, mcts_pis, mcts_vs = zip(*examples)

        # Convert to tensors
        boards = torch.tensor(np.array(boards), dtype=torch.float32)
        mcts_pis = torch.tensor(np.array(mcts_pis), dtype=torch.float32)
        mcts_vs = torch.tensor(np.array(mcts_vs), dtype=torch.float32)

        # Ensure tensors are on the correct device
        device = next(self.nnet.parameters()).device
        boards, mcts_pis, mcts_vs = boards.to(device), mcts_pis.to(device), mcts_vs.to(device)

        # Forward pass
        pi_logits, v_pred = self.nnet(boards)

        # Calculate loss for policy using KLDivLoss or equivalent for target distributions
        log_softmax_pi_logits = torch.log_softmax(pi_logits, dim=1)
        # Negative sum for KL divergence, then mean across batch
        loss_pi = -torch.sum(mcts_pis * log_softmax_pi_logits, dim=1).mean()

        # Value loss (Mean Squared Error)
        loss_v = self.criterion_v(v_pred.squeeze(-1), mcts_vs)

        total_loss = loss_pi + loss_v

        # Backward pass and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"Warning: No checkpoint found at '{filepath}'.")
            return
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded from '{filepath}'.")

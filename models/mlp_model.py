"""
MLP model for federated learning on UCI Adult dataset.
Configurable architecture: input → [hidden layers with ReLU + Dropout] → binary output.
"""
import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    Configurable MLP for tabular binary classification in federated settings.

    Args:
        input_dim:    Number of input features (read dynamically from preprocessed data).
        hidden_dims:  List of hidden layer widths. Default: [128, 64].
        dropout_rate: Dropout probability after each hidden layer. Default: 0.2.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout_rate: float = 0.2):
        super(MLPModel, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = hidden_dim
        layers += [nn.Linear(prev_dim, 1), nn.Sigmoid()]

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)


# Backward-compatibility alias
SimpleMLPModel = MLPModel


if __name__ == "__main__":
    import os
    import pickle

    print("=" * 60)
    print("TESTING MLP MODEL ARCHITECTURE")
    print("=" * 60)

    # Read input_dim from preprocessed data if available
    data_path = 'data/adult_train.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            meta = pickle.load(f)
        input_dim = meta['input_dim']
        print(f"\n📂 input_dim={input_dim} loaded from {data_path}")
    else:
        input_dim = 41
        print(f"\n⚠️  Data not found — using default input_dim={input_dim}")

    hidden_dims = [128, 64]
    dropout_rate = 0.2

    model = MLPModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)

    test_input = torch.randn(10, input_dim)
    model.eval()
    with torch.no_grad():
        output = model(test_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📐 Model Architecture:")
    print(f"   Input:  {input_dim} features")
    for i, h in enumerate(hidden_dims):
        print(f"   Hidden {i+1}: {h} neurons  (ReLU + Dropout({dropout_rate}))")
    print(f"   Output: 1 neuron (Sigmoid)")

    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters:     {total_params}")
    print(f"   Trainable parameters: {trainable_params}")

    print(f"\n✅ Forward Pass Test:")
    print(f"   Input shape:  {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("\n" + "=" * 60)
    print("✅ MODEL CREATED AND TESTED SUCCESSFULLY!")
    print("=" * 60)

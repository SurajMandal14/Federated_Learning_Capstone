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


def model_summary(model, input_dim: int, batch_size: int = 32):
    """Print a Keras-style layer-by-layer summary table."""

    # Collect layer info by tracing the Sequential network
    rows = []
    current_dim = input_dim

    for layer in model.network:
        name = layer.__class__.__name__
        if isinstance(layer, nn.Linear):
            out_dim  = layer.out_features
            params   = layer.weight.numel() + layer.bias.numel()
            rows.append((f"Linear", f"({batch_size}, {out_dim})", params))
            current_dim = out_dim
        elif isinstance(layer, nn.ReLU):
            rows.append(("ReLU", f"({batch_size}, {current_dim})", 0))
        elif isinstance(layer, nn.Dropout):
            rows.append((f"Dropout (p={layer.p})", f"({batch_size}, {current_dim})", 0))
        elif isinstance(layer, nn.Sigmoid):
            rows.append(("Sigmoid", f"({batch_size}, {current_dim})", 0))
        else:
            rows.append((name, f"({batch_size}, {current_dim})", 0))

    # Column widths
    c1, c2, c3 = 30, 20, 12
    div  = f"├{'─'*(c1+2)}┼{'─'*(c2+2)}┼{'─'*(c3+2)}┤"
    top  = f"┌{'─'*(c1+2)}┬{'─'*(c2+2)}┬{'─'*(c3+2)}┐"
    bot  = f"└{'─'*(c1+2)}┴{'─'*(c2+2)}┴{'─'*(c3+2)}┘"
    head = f"│ {'Layer (type)':<{c1}} │ {'Output Shape':<{c2}} │ {'Param #':>{c3}} │"

    print(top)
    print(head)
    print(div.replace("├", "╞").replace("┤", "╡").replace("┼", "╪").replace("─", "═"))
    for i, (layer_name, out_shape, params) in enumerate(rows):
        param_str = f"{params:,}" if params > 0 else "0"
        print(f"│ {layer_name:<{c1}} │ {out_shape:<{c2}} │ {param_str:>{c3}} │")
        if i < len(rows) - 1:
            print(div)
    print(bot)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = total * 4 / (1024 ** 2)   # float32 = 4 bytes
    print(f"\n Total params:         {total:,} ({size_mb:.2f} MB)")
    print(f" Trainable params:     {trainable:,} ({size_mb:.2f} MB)")
    print(f" Non-trainable params: {total - trainable:,} (0.00 MB)")


if __name__ == "__main__":
    import os
    import pickle

    # Read input_dim from preprocessed data if available
    data_path = 'data/adult_train.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            meta = pickle.load(f)
        input_dim = meta['input_dim']
    else:
        input_dim = 41

    hidden_dims  = [128, 64]
    dropout_rate = 0.2
    batch_size   = 32

    model = MLPModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)

    print(f'\nModel: "MLPModel"')
    model_summary(model, input_dim=input_dim, batch_size=batch_size)

    # Forward pass sanity check
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(batch_size, input_dim)
        output = model(test_input)
    print(f"\n✅ Forward pass: input {list(test_input.shape)} → output {list(output.shape)}")

import torch
from transformer_net import TransformerNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = TransformerNet()
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "saved_models", "candy.pth"
    )
    state_dict = torch.load(model_path)
    filtered_dict = {
        k: v
        for k, v in state_dict.items()
        if not (k.endswith("running_mean") or k.endswith("running_var"))
    }
    model.load_state_dict(filtered_dict, strict=False)
    model.to(device)
    model.eval()
    return model, device

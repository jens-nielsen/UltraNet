import torch

if __name__ == "__main__":

    from src.models.opno import OPNO1D

    model_name = "test"
    model = torch.load(f"./models/{model_name}.pt", weights_only=False)
    
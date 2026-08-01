import torch
import argparse

from src.models import NeuralOperatorModel, ModelType
from src.utils import DataType, NeuralOperatorDataset, Trainer


if __name__ == "__main__":
    # Give inputs, additional configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=ModelType, choices=list(ModelType), required=True, help='Model type: fno, opno, ultranet')
    parser.add_argument('--data', type=DataType, choices=list(DataType), required=True, help='Data type: burgers, darcy, helmholtz')
    parser.add_argument('--run_name', type=str, required=True, help='WandB run name for logging')
    parser.add_argument('--project_name', type=str, default="UltraNet", help='WandB project name for logging')
    parser.add_argument('--subsample', type=int, default=1, help='Subsample factor for training data')
    parser.add_argument('--arg', action='append', help='Enter items as model_arg=value')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--ntrain', type=float, default=1000, help='Number of training samples')
    parser.add_argument('--ntest', type=float, default=200, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    args = parser.parse_args()

    # Initialize dataset
    data = NeuralOperatorDataset(args.data, is_cheby=False if args.model == "fno" else True , ntrain=args.ntrain, ntest=args.ntest, batch_size=args.batch_size, subsample=args.subsample)

    # Initialize model
    model_params = {k: eval(v) for k, v in (arg.split('=') for arg in args.arg)}
    model = NeuralOperatorModel(args.model, d=data.d, **model_params)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Initialize loss function
    loss_fn = torch.nn.MSELoss()

    # Initialize device and Logging name
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Train
    trainer = Trainer(model, optimizer, loss_fn, device=device, data=data, scheduler=scheduler, args=args)
    trainer.run(num_epochs=args.num_epochs)

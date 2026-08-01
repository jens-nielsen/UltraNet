import torch
import wandb
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, data, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.data = data
        self.scheduler = scheduler
        self.args = args

    def step(self, x, y):
        out = self.model(x)
        
        # Decode the output and target using the normalizer
        out = (out * (self.data.u_std + self.data.eps)) + self.data.u_mean
        y = (y * (self.data.u_std + self.data.eps)) + self.data.u_mean

        loss = self.loss_fn(out, y)

        return loss
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        loss = self.step(x, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate_step(self, x, y):
        loss = self.step(x, y).item()
        return loss

    
    def run(self, num_epochs):

        print(self.args)
        assert False
        if self.args.wandb_project_name is not None:
            wandb.init(
                project=self.args.wandb_project_name,
                config={**vars(self.args), "nparams": self.model.count_params()},
                name=self.args.wandb_run_name,
            )

        tbar = tqdm(range(num_epochs))

        # Move model and data normalizer to device
        self.model.to(self.device)
        self.data.a_mean, self.data.a_std = self.data.a_mean.to(self.device), self.data.a_std.to(self.device)
        self.data.u_mean, self.data.u_std = self.data.u_mean.to(self.device), self.data.u_std.to(self.device)

        for epoch in tbar:

            # Train
            self.model.train()
            train_l2 = 0
            for x, y in self.data.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.train_step(x, y)
                train_l2 += loss

            self.scheduler.step()

            # Test 
            self.model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for x, y in self.data.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    loss = self.validate_step(x, y)
                    test_l2 += loss

            train_l2 /= self.data.ntrain
            test_l2 /= self.data.ntest

            tbar.set_description(
                f"Epoch {epoch}, Train L2: {train_l2:.4f}, Test L2: {test_l2:.4f}"
            )
            # if self.wandb_project_name is not None:
            #     wandb.log({"Train L2": train_l2, "Test L2": test_l2})

        if self.wandb_project_name is not None:
            wandb.finish()


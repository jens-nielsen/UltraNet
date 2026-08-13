import wandb, pandas
import argparse

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()


    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', nargs='+')


    args = parser.parse_args()
    print(args)


    api = wandb.Api()

    def m_loss(pth: list[str]):
        for p in pth:
            run = api.run(pth)
            return run.history(keys=["Test L2"])["Test L2"].min()

    print(m_loss(args.pth))

    
# Create a class parser for the arguments.
import argparse
class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.parser.add_argument(
            "--config",
            type=str,
            default="config.yaml",
            help="Path to the config file.",
        )
        self.parser.add_argument(
            "--model",
            type=str,
            default="model.pth",
            help="Path to the model file.",
        )
        self.parser.add_argument(
            "--data",
            type=str,
            default="data",
            help="Path to the data directory.",
        )
        self.parser.add_argument(
            "--output",
            type=str,
            default="output",
            help="Path to the output directory.",
        )
        self.parser.add_argument(
            "--log",
            type=str,
            default="log",
            help="Path to the log directory.",
        )
        self.parser.add_argument(
            "--mode",
            type=str,
            default="train",
            help="Mode: train, test, or eval.",
        )
        self.parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device: cuda or cpu.",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="Number of workers for data loader.",
        )

        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Debug mode: only use a small subset of the dataset.",
        )
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume training from the last checkpoint.",
        )
        self.parser.add_argument(
            "--evaluate",
            action="store_true",
            help="Evaluate the model on the test set.",
        )

        # Test options
        self.parser.add_argument("--model-pth", 
            type=str, 
            default='log/snapshot', 
            help="=encoder path"
        )
        self.parser.add_argument("--iter", 
            type=str, 
            default='40000', 
            help="=iteration number"
        )
        self.parser.add_argument("--wandb_mode", 
            type=str, 
            default="disabled", 
            help="=wandb status, online or disabled"
        )

        # Training options
        self.parser.add_argument("--exp-name", 
            type=str, 
            default=None, 
            help="=exp name"
        )
        self.parser.add_argument("--seed", 
            type=int, 
            default=888, help="=seed"
        )
        self.parser.add_argument("--weight", 
            type=float, 
            default=1.0, 
            help="=loss weight"
        )

        self.parser.add_argument("--run_name",
            type=str, 
            default=None,   
            help="insert_run_name",
            required=True
        )
        self.parser.add_argument("--model_name",
            type=str,
            default=None,
            help="expi, squeeze",
            required=True
        )


    def parse(self):
        return self.parser.parse_args()


# main
if __name__ == "__main__":

# print parser
    parser = Parser()
    args = parser.parse()
    print(args)

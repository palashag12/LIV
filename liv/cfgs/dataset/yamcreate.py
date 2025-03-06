import yaml
import os

# Define variables
dataset_name = "epickitchen"
datapath_train = "/Datasets/jasonyma/epic-kitchens-100-annotations/"
wandb_project = "epickitchen"
hydra_job_name = "train_liv_epickitchen"

# Variable for the output YAML file path
output_dir = "/home/pa1077"  # Change this to your desired directory
output_filename = "config.yaml"
output_path = os.path.join(output_dir, output_filename)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

data = {
    "# @package _global_": None,
    "dataset": dataset_name,
    "datapath_train": datapath_train,
    "wandbproject": wandb_project,
    "hydra": {
        "job": {
            "name": hydra_job_name
        }
    }
}

# Custom YAML dumper to preserve comments
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

def yaml_represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', "# @package _global_")

yaml.add_representer(type(None), yaml_represent_none)

with open(output_path, "w") as file:
    yaml.dump(data, file, Dumper=MyDumper, default_flow_style=False, sort_keys=False)

print(f"YAML file saved to: {output_path}")

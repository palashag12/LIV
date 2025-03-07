import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# TODO: replace with the absolute path of dataset!
data_path = "/home/pa1077/LIBERO/libero/datasets/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo/AGENTVIEW"
trajs = glob.glob(data_path + "/*/")

manifest = {"index": [], "directory": [], "num_frames": [], "text": []}

for i, traj in tqdm(enumerate(trajs)):
    try:
        data = np.load(os.path.join(traj, "metadata.npy"), allow_pickle=True).item()
        task = data['text']
        states = data['states']
        manifest["index"].append(i)
        manifest["directory"].append(traj)
        manifest["num_frames"].append(len(states))
        manifest["text"].append(task)
    except:
        print(f"skipped {traj} ")

manifest = pd.DataFrame(manifest)
manifest.to_csv(f"{data_path}/manifest.csv")
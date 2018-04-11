#runs simulation on all TMY files in a folder
from tmy import run
from os import listdir
import numpy as np
from tqdm import tqdm

def run_folder(folder, output_loc, steps, albedo_range = (0.2, 0.5), albedo_step=0.05):
    '''
    Run all TMY files in folder
    '''

    for tmy in tqdm(listdir(folder)):
        loc = "{}/{}".format(folder, tmy)
        name = tmy.split(".")[0]
        for albedo in tqdm(np.arange(albedo_range[0], albedo_range[1], albedo_step)):
            run(loc, albedo, output_loc, name, steps)

if __name__=="__main__":
    # folder = "../../data/alltmy3a"
    folder = "../../data/tmy_3_test"
    steps = 100
    run_folder(folder, "../../plots/all_tmy", steps)

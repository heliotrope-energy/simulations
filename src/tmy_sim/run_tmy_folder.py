#runs simulation on all TMY files in a folder
from tmy import run
from os import listdir
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def run_parallel(args):
    # print("spawning processs")
    run(*args)
    # print("finished")


def run_folder(folder, output_loc, steps, albedo_range = (0.2, 0.5), albedo_step=0.3):
    '''
    Run all TMY files in folder
    '''

    args = []
    for tmy in tqdm(listdir(folder)[0:10]):
        loc = "{}/{}".format(folder, tmy)
        name = tmy.split(".")[0]
        for albedo in np.arange(albedo_range[0], albedo_range[1], albedo_step):
            args.append((loc, albedo, output_loc, name, steps))

    with Pool(4) as p:
        for _ in tqdm(p.imap_unordered(run_parallel, args), total=len(args)):
            pass

if __name__=="__main__":
    # folder = "../../data/alltmy3a"
    folder = "../../data/alltmy3a"
    steps = 'max'
    run_folder(folder, "../../plots/all_tmy", steps)

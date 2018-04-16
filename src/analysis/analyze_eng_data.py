#analyzing energy data, determining difference between optimal and astronomical
import pandas as pd
from os import listdir
import pvlib
from tqdm import tqdm
import csv

#TODO: generate optimal behavior graphs

def summarize_folder(folder, outfile=None):
    diffs = []

    for filename in listdir(folder):
        fpath = "{}/{}".format(folder, filename)
        df = pd.read_pickle(fpath)

        df["opt_is_greater"] = df['opt_eng'] > df['astro_eng']

        df["pct_change"] = df['opt_eng']/df['astro_eng']*100 - 100

        df["pct_change"].fillna(0, inplace=True)
        astro_energy = df['astro_eng'].sum()
        opt_energy = df['opt_eng'].sum()

        pct_change = (opt_energy/astro_energy)*100 - 100

        diffs.append((filename, pct_change))

    biggest_pct = max(diffs, key=lambda tup: tup[1])

    print("largest percent change: {}".format(biggest_pct))

    diffs.sort(key=lambda tup: tup[1], reverse=True)

    if outfile:
        outstrings = ["{}: {}".format(tup[0], tup[1]) for tup in diffs]

        with open(outfile, 'w') as f:
            f.write("\n".join(outstrings))

    return diffs


def match_with_tmy(data, tmy_folder, outfile, tmy_pref="tmy"):
    '''
    Matches with corresponding TMY values
    '''

    info_all = []

    for dat in tqdm(data):
        dat_num = dat[0].replace(".pkl", "").split("-")[-1]

        fname = "{}/{}.{}".format(tmy_folder, tmy_pref, dat_num)

        #getting metadata
        _, meta = pvlib.tmy.readtmy3(filename=fname)

        sand_point = pvlib.location.Location(meta['latitude'], meta['longitude'], tz=meta['TZ'], altitude=meta['altitude'], name=meta['Name'].replace('"',''))

        info = meta['latitude'], meta['longitude'], meta['Name'], meta['State']


        info_all.append({"lat":meta['latitude'], "long": meta['longitude'], "name": meta['Name'], "state": meta['State'], "pct_increase": dat[1]})


    #save to CSV

    with open('outfile.csv', 'w') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=['lat', 'long', 'name', 'state', 'pct_increase'])
        writer.writeheader()
        writer.writerows(info_all)





if __name__=="__main__":
    diffs = summarize_folder("../../processed/grid_res", "summary.txt")

    match_with_tmy(diffs, "../../data/renamed", "summary.csv")

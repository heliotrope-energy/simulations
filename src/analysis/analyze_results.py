#analyzing results

from os import listdir

def load_results(folder, output):
    '''
    Summarize all TXT files in a folder
    '''
    data = []
    for file in listdir(folder):
        if file.endswith(".txt"):
            filepath="{}/{}".format(folder, file)
            with open(filepath, 'r') as f:
                lines = f.readlines()
                data_pt = {"tmy_id":lines[0], "albedo": float(lines[1]), "steps": float(lines[2])}
                for line in lines[3:]:
                    data_pt[line.split()[3]] = float(line.split()[0])
                data.append(data_pt)

    return data




if __name__=="__main__":
    results = load_results("../../plots/all_tmy", "../../plots/all_tmy_summary.txt")

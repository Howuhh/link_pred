import os
import numpy as np

PATH = os.getcwd()

# А нужно ли вообще тут так заморачиваться с путями?


def clean_graph(load_name, save_name=None):
    load_path = os.path.join(PATH, load_name)

    if save_name is None:
        save_path = load_path
    else:
        save_path = os.path.join(PATH, save_name)

    with open(load_path, "r") as raw_file:
        edge_list = raw_file.readlines()
        with open(save_path, "w") as prep_file:
            for edge in edge_list[2:]:
                new_str = " ".join(edge.split()[:2])
                prep_file.write(new_str + "\n")


def drop_parallels(load_name, save_name=None):
    load_path = os.path.join(PATH, load_name)

    if save_name is None:
        save_path = load_path
    else:
        save_path = os.path.join(PATH, save_name)
    
    # yeah, but it works
    edge_list = np.loadtxt(load_path)
    edge_list_uni = np.unique(edge_list, axis=0)
    np.savetxt(save_path, edge_list_uni, delimiter=" ", fmt='%i')


if __name__ == "__main__":
    print("Start cleaning...")
    clean_graph("data/fb-wosn-friends.edges")
    print("Droping parallels...")
    drop_parallels("data/fb-wosn-friends.edges", "data/fb_friends_uni.edges")
    print("Done cleaning!")

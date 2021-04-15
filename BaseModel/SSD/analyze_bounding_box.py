import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

f = open("./datasets/tdt4265/labels.json", 'r')
bb_file = json.load(f)
json_dict = dict(bb_file)
df_list = []
for i in range(4):
    df_list.append(pd.DataFrame({"width": [], "height": []}))

N = 100
json_length = len(json_dict["annotations"][:])
num_intervals = json_length//N
indexes = [N*i for i in range(num_intervals)]
i = 0
for index in tqdm(indexes[:-1]):
    ims = json_dict["annotations"][index:indexes[i+1]]
    category_wh = np.zeros((4, 2))
    category_counter = np.zeros(4)
    i += 1
    for im in ims:
        _, _, w, h = im["bbox"]
        category = int(im["category_id"])
        category_wh[category] += np.array([w, h])
        category_counter[category] += 1
    category_wh = (category_wh.T/category_counter).T
    for category in range(4):
        df_list[category] = df_list[category].append(
            {"width": category_wh[category, 0], "height": category_wh[category, 1]}, ignore_index=True)

for i, df in enumerate(df_list):
    print("-----------------------------------------------------------")
    print(df.describe())
    print("Aspect_ratio for class"+str(i)+": ", np.mean(df["width"]/df["height"]))
    std_h = np.std(df["height"])
    std_w = np.std(df["width"])
    print("Height within 3 std for class"+str(i)+": ", np.mean(df["height"]+3*std_h))
    print("Width within 3 std for class"+str(i)+": ", np.mean(df["width"]+3*std_w))

    fig, ax = plt.subplots(3)
    df.plot.hist(ax=ax[0], alpha=0.5, bins=40)
    df.plot.scatter(x="width", y="height", ax=ax[1], alpha=0.5)
    aspect_df = pd.DataFrame({"aspect_ratio": df["width"]/df["height"]})
    aspect_df.plot.hist(ax=ax[2], alpha=0.5, bins=80)
    for a in ax:
        a.grid()
    plt.show()

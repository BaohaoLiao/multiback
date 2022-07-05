import os

if __name__ == "__main__":
    data_dir = "../data/small_task2_filt"  # TODO: change to the directory where you save raw data for small task #1
    save_dir = "../data/small_task2_filt_concat" # TODO: change to the directory where you want to save for small task #1

    prefix = []
    lang_pairs = []
    for file in os.listdir(data_dir):
        prefix.append(".".join(file.split(".")[:2]))
        lang_pairs.append(file.split(".")[1])
    prefix = list(set(prefix))
    lang_pairs = list(set(lang_pairs))

    for i in prefix:
        src_file = os.path.join(data_dir, i + "." + i.split(".")[1].split("-")[0])
        tgt_file = os.path.join(data_dir, i + "." + i.split(".")[1].split("-")[1])
        key = i.split(".")[1]
        try:
            with open(src_file) as fp:
                src = fp.read()
            with open(tgt_file) as fp:
                tgt = fp.read()
        except:
            with open(src_file + ".filt") as fp:
                src = fp.read()
            with open(tgt_file + ".filt") as fp:
                tgt = fp.read()

        # Save
        with open(os.path.join(save_dir, "train." + key + "." + key.split("-")[0]), 'a') as fp:
            fp.write(src + "\n")
        with open(os.path.join(save_dir, "train." + key + "." + key.split("-")[1]), 'a') as fp:
            fp.write(tgt + "\n")

    data_dict = {}
    for i in lang_pairs:
        data_dict[i] = ["", ""]
    for i in prefix:
        src_file = os.path.join(data_dir, i + "." + i.split(".")[1].split("-")[0])
        tgt_file = os.path.join(data_dir, i + "." + i.split(".")[1].split("-")[1])
        key = i.split(".")[1]
        try:
            with open(src_file) as fp:
                src = fp.read()
            with open(tgt_file) as fp:
                tgt = fp.read()
        except:
            with open(src_file + ".filt") as fp:
                src = fp.read()
            with open(tgt_file + ".filt") as fp:
                tgt = fp.read()
        data_dict[key][0] += src
        data_dict[key][0] += '\n'
        data_dict[key][1] += tgt
        data_dict[key][1] += '\n'

    for k, v in data_dict.items():
        with open(os.path.join(save_dir, "train." + k + "." + k.split("-")[0]), 'w') as fp:
            fp.write(v[0])
        with open(os.path.join(save_dir, "train." + k + "." + k.split("-")[1]), 'w') as fp:
            fp.write(v[1])



from pathlib import Path
# input is r, e1, e2, ...
# transform the format to e1, r, e2, r_2, e3, r_3, e4 ...

def transform_one_folder(input_folder, output_folder):
    # read train.txt, valid.txt, test.txt and aux.txt, transform them into the format we want
    # write them into output_folder

    # read original entity name
    id2entity = {}
    id2relation = {}
    with open(input_folder / "entities.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # substitute the "," in entity name with "--"
            id2entity[line[0]] = line[1].replace(",", "--")
    with open(input_folder / "relations.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split('\t')

            id2relation[line[0]] = line[1].replace(",", "--")

    # input is r, e1, e2, ..., where the delimiter is "\t"
    # transform the format to e1, r, e2, r_2, e3, r_3, e4 ..., where the delimiter is ","
    target_txts = ["train.txt", "valid.txt", "test.txt", "aux.txt"]
    for target_txt_i in target_txts:
        output_txt = output_folder / target_txt_i
        f_output = open(output_txt, "w")
        with open(input_folder / target_txt_i, "r") as f:
            for line in f.readlines():
                row = line.strip().split('\t')
                r = id2relation[row[0]]
                h = id2entity[row[1]]
                t = id2entity[row[2]]
                other_entities = [id2entity[row[i]] for i in range(3, len(row))]
                main_triplet_str = h + "," + r + str(1) + "," + t
                qualifier_str = ""
                for i, other_entity_i in enumerate(other_entities):
                    qualifier_str += "," + r + str(i+2) + "," + other_entity_i
                f_output.write(main_triplet_str + qualifier_str + "\n")
        f_output.close()

    # examine if train.txt contains all entities and relations appeared
    train_entities = set()
    train_relations = set()
    with open(input_folder / "train.txt", "r") as f:
        for line in f.readlines():
            row = line.strip().split('\t')
            train_relations.add(row[0])
            train_entities.update(row[1:])
    target_txts = ["valid.txt", "test.txt", "aux.txt"]
    for target_txt_i in target_txts:
        print(target_txt_i)
        entities = set()
        relations = set()
        with open(input_folder / target_txt_i, "r") as f:
            for line in f.readlines():
                row = line.strip().split('\t')
                relations.add(row[0])
                # entities.update(row[1:])
                entities.update(row[1:3])
        # print entities/relations that are not in train.txt
        print("entities in", target_txt_i, "but not in train.txt:", entities - train_entities)
        print("relations in", target_txt_i, "but not in train.txt:", relations - train_relations)


# please run me in the parent directory of the folder "script"
if __name__ == "__main__":
    input_root_path = Path("./data/GMPNN_data")

    output_path = Path("./data/GMPNN_data_ours")
    output_path.mkdir(exist_ok=True)

    # iterate sub_folder
    for sub_folder in input_root_path.iterdir():
        (output_path / sub_folder.name).mkdir(exist_ok=True)
        transform_one_folder(sub_folder, output_path / sub_folder.name)


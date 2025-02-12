if __name__ == "__main__":
    all_txt = ""
    for fid in range(1, 121):
        with open(f"output/{fid}.txt", "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    all_txt += line
            all_txt += "\n"

    with open("hlm_all.txt", "w", encoding="utf-8") as file:
        file.write(all_txt)

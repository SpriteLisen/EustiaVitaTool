import csv

line_break_tipping_point = 30

if __name__ == "__main__":
    csv_path = "game_script/translate.csv"

    need_modify_count = 0
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f)
        line_count = 1
        for row in reader:
            if len(row) >= 2:
                src, tgt = row[0], row[1]
                if len(tgt) == line_break_tipping_point:
                    need_modify_count += 1
                    print(f"换行卡在临界点 ({line_count}) ==> {tgt}")

            line_count += 1

    print('\n' + '*' * 20 + f"共计 {need_modify_count} 条 待修复" + '*' * 20)
import re
from pathlib import Path

output_dir = "extract_text"


def parse_ons_script(ons_script_path):
    file_path = Path(ons_script_path)

    output_path = file_path.with_name(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_path / file_path.name
    output_file = open(output_file_path, "w", encoding="utf-8")

    # 全屏一行渲染的文案内容
    screen_msgs = []
    # 单独的对白文案
    says = []
    # 人物名称
    persons = []
    # 人物对白文案
    person_says = []
    # 选择分支文案
    selects = []

    # 1. b__screenmsg: 抓取第一个 "" 中的内容
    re_screen_msg = re.compile(r'^b__screenmsg.*?"(.*?)"')
    # 2. b__say: 抓取第一个 "" 中的内容
    re_say = re.compile(r'^b__say(?!_)\S*\s*"(.*?)"')
    # 3. b__say_1: 抓取两个 "" 中的内容
    re_say1 = re.compile(r'^b__say_1.*?"(.*?)".*?"(.*?)"')
    # 4. b_f_164: 抓取分支中的内容

    with open(file_path, "r", encoding="gb2312", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # 匹配 b__screenmsg
            m1 = re_screen_msg.match(line)
            if m1:
                screen_msg = m1.group(1)
                screen_msgs.append(screen_msg)

                output_file.write(screen_msg + "\n")
                continue

            # 匹配 b__say（排除 b__say_1）
            m2 = re_say.match(line)
            if m2:
                say = m2.group(1)
                says.append(say)

                output_file.write(say + "\n")
                continue

            # 匹配 b__say_1
            m3 = re_say1.match(line)
            if m3:
                person_say = m3.group(1)
                person_name = "【" + m3.group(2) + "】"
                person_says.append(person_say)  # 对白
                persons.append(person_name)  # 人物名

                output_file.write(person_name + "\n")
                output_file.write(person_say + "\n")
                continue

            # 匹配 b_f_164 分支选择
            matches = re.findall(r'^b_f_164\s*("[^"]*"(?:\s*,\s*"[^"]*")*)', line, re.DOTALL)
            matches = [m.strip('"') for m in matches if m and m != 'b_f_164']
            if matches:
                for m in matches:
                    selects.append(m)
                continue

    output_file.close()

    # 输出测试
    # print("screenmsg:", screen_msgs)
    # print("says:", says)
    # print("persons:", persons)
    # print("person_says:", person_says)

    # 合并所有文本
    all_text = "".join(screen_msgs + says + persons + person_says + selects)

    # 统计字符集合
    # \u3000 是全角的空格
    unique_chars = sorted(set(all_text))

    # 输出
    print("总字符数:", len(unique_chars))
    print("字符列表:", unique_chars)

    print("分支列表:", selects)


def parse_pymo_script(pymo_script_path):
    file_path = Path(pymo_script_path)

    output_path = file_path.with_name(output_dir)
    output_file_path = output_path / "pymo_version.txt"
    output_file = open(output_file_path, "w", encoding="utf-8")

    # 全屏一行渲染的文案内容
    screen_msgs = []
    # 单独的对白文案
    says = []
    # 人物名称
    persons = []
    # 人物对白文案
    person_says = []
    # 选择分支文案
    selects = []

    # 1. b__screenmsg: 抓取第一个 "" 中的内容
    re_screen_msg = re.compile(r'^b__screenmsg.*?"(.*?)"')
    # 2. #say content || #say 【name】,「content」
    re_say = re.compile(r'^#say\s+([^,\s]+)(?:,([^,\s]+))?')
    # 3. #sel n
    re_select = re.compile(r'^#sel\s+(\d+)')

    # 解析实际内容
    def parse_content(files):
        for main_script in files:
            with open(main_script, "r", encoding="utf8", errors="ignore") as f:
                in_case = False
                now_select_count = 0
                now_select_index = 0

                for line in f:
                    line = line.strip()

                    # 解析分支文案
                    select_match = re_select.match(line)
                    if select_match:
                        in_case = True
                        now_select_count = int(select_match.group(1))
                        continue

                    if in_case:
                        now_select_index += 1
                        selects.append(line)
                        if now_select_index >= now_select_count:
                            now_select_index = 0
                            now_select_count = 0
                            in_case = False
                        continue

                    # 匹配 b__screenmsg
                    m1 = re_screen_msg.match(line)
                    if m1:
                        screen_msg = m1.group(1)
                        screen_msgs.append(screen_msg)

                        output_file.write(screen_msg + "\n")
                        continue

                    m2 = re_say.match(line)
                    if m2:
                        # 匹配 #say 【name】,「content」
                        if m2.group(2):
                            person_name = m2.group(1)
                            person_say = m2.group(2)
                            person_says.append(person_say)  # 对白
                            persons.append(person_name)  # 人物名

                            output_file.write(person_name + "\n")
                            output_file.write(person_say + "\n")
                            continue
                        # 匹配 #say content
                        else:
                            say = m2.group(1)
                            says.append(say)

                            output_file.write(say + "\n")
                            continue

    # 解析主剧情
    main_script_path = file_path / "main"
    parse_content(main_script_path.glob('*.[Tt][Xx][Tt]'))

    # 解析附录剧情
    appendix_script_path = file_path / "appendix"
    parse_content(appendix_script_path.glob('*.[Tt][Xx][Tt]'))

    # 解析终末话语
    parse_content(file_path.glob('*.[Tt][Xx][Tt]'))

    output_file.close()

    # 合并所有文本
    all_text = "".join(screen_msgs + says + persons + person_says + selects)

    # 统计字符集合
    # \u3000 是全角的空格
    unique_chars = sorted(set(all_text))

    # 输出
    print("总字符数:", len(unique_chars))
    print("字符列表:", unique_chars)

    print("分支列表:", selects)


if __name__ == "__main__":
    parse_ons_script("game_script/ons_version.txt")
    parse_pymo_script("game_script/pymo")

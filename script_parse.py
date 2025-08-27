import re
from pathlib import Path
from opencc import OpenCC

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

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
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
            matches = re.match(r'^b_f_164\s*("[^"]*"(?:\s*,\s*"[^"]*")*)', line)
            if matches:
                options = re.findall(r'"([^"]*)"', matches.group(1))
                for m in options:
                    selects.append(m)
                continue

    output_file.close()

    # 输出测试
    # print("screenmsg:", screen_msgs)
    # print("says:", says)
    # print("persons:", sorted(set(persons)))
    # print("person_says:", person_says)

    # 合并所有文本
    all_text = "".join(screen_msgs + says + persons + person_says + selects)

    # 统计字符集合
    # \u3000 是全角的空格
    unique_chars = sorted(set(all_text))

    # 输出
    # print("总字符数:", len(unique_chars))
    # print("字符列表:", unique_chars)

    # print("分支总数:", len(selects))
    # print("分支列表:", selects)


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
                        output_file.write(line + "\n")
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
    parse_content(
        sorted(main_script_path.glob('*.[Tt][Xx][Tt]'), key=lambda x: x.name.lower())
    )

    # 解析附录剧情
    appendix_script_path = file_path / "appendix"
    parse_content(
        sorted(appendix_script_path.glob('*.[Tt][Xx][Tt]'), key=lambda x: x.name.lower())
    )

    # 解析终末话语
    parse_content(
        sorted(file_path.glob('*.[Tt][Xx][Tt]'), key=lambda x: x.name.lower())
    )

    output_file.close()

    # 输出测试
    # print("screenmsg:", screen_msgs)
    # print("says:", says)
    print("persons:", sorted(set(persons)))
    # print("person_says:", person_says)

    # 合并所有文本
    all_text = "".join(screen_msgs + says + persons + person_says + selects)

    # 统计字符集合
    # \u3000 是全角的空格
    unique_chars = sorted(set(all_text))

    # 输出
    print("总字符数:", len(unique_chars))
    print("字符列表:", unique_chars)

    cc = OpenCC('t2s')  # 繁体 -> 简体
    traditional = []
    for ch in unique_chars:
        if cc.convert(ch) != ch:  # 转换后不一样，说明是繁体字
            traditional.append(ch)
    print("繁体字:", traditional)

    print("分支总数:", len(selects))
    print("分支列表:", selects)


def parse_psv_script(psv_script_path):
    file_path = Path(psv_script_path)

    output_path = file_path.with_name(output_dir)

    output_file_path = output_path / "psv_version.txt"
    output_file = open(output_file_path, "w", encoding="utf-8")

    # 全屏一行渲染的文案内容
    screen_msgs = []
    # 对白文案
    says = []
    # 人物名称
    persons = []
    # 选择分支文案
    selects = []

    # 定义正则表达式模式
    rtth_pattern = re.compile(r'!_RTTH\([^,]*,([^)]*)\)\)')
    zmyyyy_pattern = re.compile(r'<\w+>_ZM\w+\(([^)]*)\)')
    mtlk_pattern = re.compile(r'!_MTLK\([^,]*,\s*([^)]*)\)')
    selr_pattern = re.compile(r'<\w+>_SELR\([^;]*;/([^)]*)\)\)')

    files = sorted(file_path.glob('*.[Tt][Pp][Ll]'), key=lambda x: x.name.lower())
    for psv_script in files:
        # entry_200 是测试文本
        if psv_script.name == "entry_200.tpl":
            continue

        with open(psv_script, "r", encoding="shift_jis", errors="ignore") as f:
            for line in f:
                # 文本中有一些部分在结尾会多出一个 . 符号
                line = line.strip().replace("」.", "」")

                # 1. 解析整行文案
                if line.startswith('!_RTTH'):
                    match = rtth_pattern.search(line)
                    if match:
                        screen_msg = match.group(1)
                        screen_msgs.append(screen_msg)
                        output_file.write(screen_msg + "\n")

                # 2. 解析文本内容
                elif '_ZM' in line and '(' in line and ')' in line:
                    match = zmyyyy_pattern.search(line)
                    if match:
                        say = match.group(1)
                        says.append(say)
                        output_file.write(say + "\n")

                # 3. 解析人名
                elif line.startswith('!_MTLK'):
                    match = mtlk_pattern.search(line)
                    if match:
                        person = match.group(1)
                        persons.append(person)
                        output_file.write(person + "\n")

                # 4. 解析分支内容
                elif '_SELR(' in line:
                    match = selr_pattern.search(line)
                    if match:
                        select = match.group(1)
                        selects.append(select)
                        output_file.write(select + "\n")

    output_file.close()

    # 输出测试
    # print("screenmsg:", screen_msgs)
    # print("says:", says)
    # print("persons:", sorted(set(persons)))

    # 合并所有文本
    all_text = "".join(screen_msgs + says + persons + selects)

    # 统计字符集合
    # \u3000 是全角的空格
    unique_chars = sorted(set(all_text))

    # 输出
    print("总字符数:", len(unique_chars))
    print("字符列表:", unique_chars)

    # print("分支总数:", len(selects))
    # print("分支列表:", selects)


if __name__ == "__main__":
    parse_ons_script("game_script/ons_version.txt")
    print("----------------------------------------------------------------------------------------------------------")
    parse_pymo_script("game_script/pymo")
    print("----------------------------------------------------------------------------------------------------------")
    parse_psv_script("game_script/psv")

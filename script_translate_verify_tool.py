import csv

# 人名对照表
check_map = {
    "【不蝕金鎖の男】": "【不蚀金锁成员】",
    "【中年の男】": "【中年乞丐】",
    "【女の主】": "【主人】",
    "【坊主の男】": "【光头男人】",
    "【ラング副隊長】": "【兰格副队长】",
    "【ラング】": "【兰格】",
    "【カイム】": "【凯伊姆】",
    "【カイム＆ジーク】": "【凯伊姆＆吉克】",
    "【ジーク】": "【吉克】",
    "【ただのジーク】": "【吉克的声音】",
    "【中年の見物客】": "【围观的中年人】",
    "【派手な女見物客】": "【围观的女人】",
    "【聖女】": "【圣女】",
    "【太った羽狩り】": "【壮实的羽狩】",
    "【太鼓腹の酔客】": "【大肚子的醉汉】",
    "【ヒゲの酔客】": "【大胡子醉汉】",
    "【オズ】": "【奥兹】",
    "【女】": "【女】",
    "【女性の声】": "【女声】",
    "【店主】": "【店主】",
    "【少女】": "【少女】",
    "【少女の声】": "【少女声】",
    "【若い男】": "【年轻人】",
    "【クローディア】": "【库罗蒂雅】",
    "【柄の悪い見物客】": "【惊慌的观众】",
    "【羽が折れた少女】": "【折翼的少女】",
    "【メルト】": "【梅尔特】",
    "【鼻血を流している男】": "【流着鼻血的男人】",
    "【痩せた羽狩り】": "【消瘦的羽狩】",
    "【フィオネ隊長】": "【菲奥奈队长】",
    "【フィオネ】": "【菲奥奈】",
    "【覆面の男】": "【蒙面男人】",
    "【薄汚れた男】": "【邋遢的男人】",
    "【部下】": "【部下】",
    "【酔客】": "【醉酒的顾客】",
    "【アイリス】": "【阿伊莉斯】",
    "【付き人】": "【随从】",
    "【高い男の声】": "【高挑的男人】",
    "【ルキウス】": "【鲁基乌斯】",
    "【鷲鼻の羽狩り】": "【鹰钩鼻的羽狩】",
    "【？？】": "【？？】",
    "【エリス】": "【艾莉斯】",
    "【神官】": "【神官】",
    "【ゴツい見物客】": "【粗鲁的观众】",
    "【男】": "【男人】",
    "【羽つきの少年】": "【羽化病患少年】",
    "【羽狩りの指揮者】": "【羽狩的队长】",
    "【赤毛の羽狩り】": "【红发的羽狩】",
    "【羽狩りの副隊長】": "【羽狩的副队长】",
    "【羽狩りの隊長】": "【羽狩的队长】",
    "【リサ】": "【莉莎】",
    "【太った男】": "【胖男人】",
    "【ティア】": "【缇娅】",
    "【太い男の声】": "【肥胖的男人】",
    "【男の声】": "【男声】",
    "【リシア】": "【莉西亚】",
    "【猫背の男】": "【驼背男人】",
    "【太った女】": "【胖女人】",
    "【神経質そうな女】": "【神经质的女人】",
    "【薄汚いガキ】": "【邋遢的小鬼】",
    "【痩せた老婆】": "【瘦弱的婆婆】",
    "【女の声】": "【女声】",
    "【中年の野次馬】": "【中年的起哄者】",
    "【ゴツい野次馬】": "【粗鲁的起哄者】",
    "【黒羽】": "【黑羽】",
    "【不蝕金鎖の若い男】": "【不蚀金锁的年轻人】",
    "【羽狩りたち】": "【羽狩们】",
    "【不蝕金鎖の幹部たち】": "【不蚀金锁的干部们】",
    "【クーガー】": "【库格尔】",
    "【若い羽狩り】": "【年轻的羽狩】",
    "【ルキウス卿】": "【鲁基乌斯卿】",
    "【副官】": "【副官】",
}

csv_file = "game_script/translate.csv"

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader, start=1):
        original = row[0].strip()
        translation = row[1].strip()

        if translation is None or translation == "":
            print(f"无更多内容 ==> 行号: {i}")
            break

        # 规则 1: 【xxx】
        if original.startswith("【") and original.endswith("】"):
            value = check_map.get(original, "")
            if value == "":
                print(f"行 {i} 异常, {original} 不存在")
                print(f"{original} <==> {translation}")
                break
            if translation != value:
                print(f"行 {i} 异常 【xxx】规则")
                print(f"{original} <==> {translation}")
                break
            continue

        # 规则 2: 「xxx」
        if original.startswith("「") and original.endswith("」"):
            if not (translation.startswith("「") and translation.endswith("」")):
                print(f"行 {i} 异常 「xxx」规则")
                print(f"{original} <==> {translation}")
                break
            continue

        # 规则 3: @w9@h9 或 @w1@h1 开头
        if original.startswith("@w9@h9") or original.startswith("@w1@h1"):
            if not (translation.startswith("@w9@h9") or translation.startswith("@w1@h1")):
                print(f"行 {i} 异常 @w开头规则")
                print(f"{original} <==> {translation}")
                break
            # 后面跟着「xxx」，再检查
            rest_original = original.split("@", 3)[-1]  # 取后半部分
            rest_translation = translation.split("@", 3)[-1]
            if (rest_original.startswith("「") and rest_original.endswith("」")):
                if not (rest_translation.startswith("「") and rest_translation.endswith("」")):
                    print(f"行 {i} 异常 @w后跟「xxx」规则")
                    print(f"{original} <==> {translation}")
                    break
            continue

        # 规则 4: 其他情况
        if (
                (not original.startswith("【") and not original.endswith("】") and not translation.startswith(
                    "【") and not translation.endswith("】"))
                and
                (not original.startswith("「") and not original.endswith("」") and not translation.startswith(
                    "「") and not translation.endswith("」"))
                # and
                # (not original.startswith("『") and not original.endswith("』") and not translation.startswith(
                #     "『") and not translation.endswith("』"))
        ):
            continue
        else:
            print(f"行 {i} 异常 其他规则")
            print(f"{original} <==> {translation}")
            break

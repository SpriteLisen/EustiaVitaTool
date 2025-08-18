# pip install pandas sentence-transformers scikit-learn jieba tqdm torch transformers sentencepiece

import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import hashlib
import torch
from multiprocessing import cpu_count
import platform
import time
from transformers import MarianMTModel, MarianTokenizer

# ------------------- 配置 -------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and platform.system() == "Darwin":
    device = "mps"
else:
    device = "cpu"

print(f"使用设备: {device} ({platform.system()} {platform.machine()})")

torch.set_num_threads(cpu_count())
torch.set_num_interop_threads(cpu_count())

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

CACHE_DIR = "translation_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
BATCH_SIZE = 64 if device != "mps" else 32
THRESHOLD = 0.82
TOP_K = 5
SEARCH_BATCH = 512  # 分批处理 PSV 向量
TRANSLATE_BATCH = 16  # MarianMT 翻译批量大小


# ------------------- Sentence-BERT 模型加载 -------------------
def load_sbert_model():
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)
    model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device)
    return model


sbert_model = load_sbert_model()
print("Sentence-BERT 模型加载成功")


# ------------------- MarianMT 模型加载 -------------------
def load_marian_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizer, model


# 日语->英语
ja_en_tokenizer, ja_en_model = load_marian_model("ja", "en")
# 英语->中文
en_zh_tokenizer, en_zh_model = load_marian_model("en", "zh")

print("MarianMT 两步翻译模型加载成功")


# ------------------- 缓存 -------------------
def get_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


class TranslationCache:
    def __init__(self):
        self.cache_file = os.path.join(CACHE_DIR, "translations.csv")
        try:
            self.df = pd.read_csv(self.cache_file)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=['hash', 'ja_text', 'zh_text'])
        self.total_translate_time = 0.0  # 新增统计翻译总耗时

    def get_translation(self, ja_text):
        text_hash = get_text_hash(ja_text)
        cached = self.df[self.df['hash'] == text_hash]
        if not cached.empty:
            return cached.iloc[0]['zh_text']
        return None

    def add_translation(self, ja_text, zh_text):
        text_hash = get_text_hash(ja_text)
        new_row = pd.DataFrame([[text_hash, ja_text, zh_text]], columns=['hash', 'ja_text', 'zh_text'])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.cache_file, index=False, encoding='utf-8-sig')

    def translate_japanese(self, ja_texts):
        """两步 MarianMT: 日语 -> 英语 -> 中文"""
        start_translate = time.perf_counter()  # 记录翻译开始时间

        # Step 1: 日语 -> 英语
        en_texts = []
        for i in range(0, len(ja_texts), TRANSLATE_BATCH):
            batch = ja_texts[i:i + TRANSLATE_BATCH]
            inputs = ja_en_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                translated = ja_en_model.generate(**inputs, max_length=256)
            batch_en = [ja_en_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            en_texts.extend(batch_en)

        # Step 2: 英语 -> 中文
        zh_texts = []
        for i in range(0, len(en_texts), TRANSLATE_BATCH):
            batch = en_texts[i:i + TRANSLATE_BATCH]
            inputs = en_zh_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                translated = en_zh_model.generate(**inputs, max_length=256)
            batch_zh = [en_zh_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            zh_texts.extend(batch_zh)

        end_translate = time.perf_counter()
        self.total_translate_time += (end_translate - start_translate)  # 累加翻译耗时
        return zh_texts


# ------------------- 文件加载 -------------------
def load_files():
    psv_df = pd.read_csv('game_script/extract_text/psv_version.txt', sep='\t', header=None, names=['ja_text'],
                         dtype=str)
    ons_df = pd.read_csv('game_script/extract_text/ons_version.txt', sep='\t', header=None, names=['zh_text'],
                         dtype=str)
    pymo_df = pd.read_csv('game_script/extract_text/pymo_version.txt', sep='\t', header=None, names=['zh_text'],
                          dtype=str)
    print(f"PSV {len(psv_df)}行, ONS {len(ons_df)}行, PYMO {len(pymo_df)}行")
    return psv_df, ons_df, pymo_df


# ------------------- 编码 -------------------
def encode_texts(texts, cache_file):
    if os.path.exists(cache_file):
        embeddings = torch.load(cache_file, map_location=device)
    else:
        embeddings = sbert_model.encode(texts, batch_size=BATCH_SIZE, device=device, convert_to_tensor=True)
        torch.save(embeddings, cache_file)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} 分钟"
    else:
        return f"{seconds / 3600:.2f} 小时"


# ------------------- 主逻辑 -------------------
def main():
    translation_cache = TranslationCache()
    psv_df, ons_df, pymo_df = load_files()

    ons_texts = [str(t).strip() for t in ons_df['zh_text'].tolist() if pd.notna(t)]
    pymo_texts = [str(t).strip() for t in pymo_df['zh_text'].tolist() if pd.notna(t)]
    ja_texts = [str(t).strip() for t in psv_df['ja_text'].tolist()]

    print("编码 ONS 文本向量...")
    ons_embeddings_file = os.path.join(CACHE_DIR, "ons_embeddings.pt")
    ons_embeddings = encode_texts(ons_texts, ons_embeddings_file).to(device)

    print("编码 PYMO 文本向量...")
    pymo_embeddings_file = os.path.join(CACHE_DIR, "pymo_embeddings.pt")
    pymo_embeddings = encode_texts(pymo_texts, pymo_embeddings_file).to(device)

    print("编码 PSV 文本向量...")
    psv_embeddings_file = os.path.join(CACHE_DIR, "psv_embeddings.pt")
    psv_embeddings = encode_texts(ja_texts, psv_embeddings_file).to(device)

    results = []
    start_time = time.perf_counter()
    pbar = tqdm(total=len(ja_texts), desc="处理进度")

    for start in range(0, len(ja_texts), SEARCH_BATCH):
        end = min(start + SEARCH_BATCH, len(ja_texts))
        psv_batch = psv_embeddings[start:end]

        if len(ons_texts) > 0:
            sim_ons = torch.matmul(psv_batch, ons_embeddings.T)
            topk_sim_ons, topk_idx_ons = torch.topk(sim_ons, k=TOP_K, dim=1)
        else:
            topk_sim_ons = torch.zeros((end - start, TOP_K), device=device)
            topk_idx_ons = torch.zeros((end - start, TOP_K), dtype=torch.long, device=device)

        if len(pymo_texts) > 0:
            sim_pymo = torch.matmul(psv_batch, pymo_embeddings.T)
            topk_sim_pymo, topk_idx_pymo = torch.topk(sim_pymo, k=TOP_K, dim=1)
        else:
            topk_sim_pymo = torch.zeros((end - start, TOP_K), device=device)
            topk_idx_pymo = torch.zeros((end - start, TOP_K), dtype=torch.long, device=device)

        for i in range(end - start):
            ja_text = ja_texts[start + i]
            best_match = None
            best_source = None
            best_sim = 0.0

            if len(ons_texts) > 0 and topk_sim_ons[i, 0] >= THRESHOLD:
                best_match = ons_texts[topk_idx_ons[i, 0]]
                best_source = "ons"
                best_sim = float(topk_sim_ons[i, 0])

            if len(pymo_texts) > 0 and best_sim < 0.93 and topk_sim_pymo[i, 0] >= THRESHOLD:
                best_match = pymo_texts[topk_idx_pymo[i, 0]]
                best_source = "pymo"
                best_sim = float(topk_sim_pymo[i, 0])

            if not best_match:
                cached_trans = translation_cache.get_translation(ja_text)
                if cached_trans:
                    best_match = cached_trans
                    best_source = "translate(cached)"
                else:
                    zh_text = translation_cache.translate_japanese([ja_text])[0]
                    translation_cache.add_translation(ja_text, zh_text)
                    best_match = zh_text
                    best_source = "translate"

            results.append({
                'original': ja_text,
                'translation': best_match,
                'source': best_source,
                'similarity': best_sim if best_source in ['ons', 'pymo'] else None
            })
            pbar.update(1)

    pbar.close()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"GPU/CPU 比对耗时: {format_time(elapsed)}")
    print(f"翻译总耗时: {format_time(translation_cache.total_translate_time)}")  # 新增输出翻译耗时

    pd.DataFrame(results).to_csv('final_translations.csv', index=False, encoding='utf-8-sig')
    print("处理完成，结果已保存到 final_translations.csv")


if __name__ == "__main__":
    main()

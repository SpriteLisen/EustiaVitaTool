# pip install pandas sentence-transformers scikit-learn jieba tqdm torch transformers

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
import hashlib
import torch
from multiprocessing import cpu_count
import platform

# ------------------- 配置 -------------------
# 自动检测设备类型
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and platform.system() == "Darwin":
    device = "mps"  # Apple Metal
else:
    device = "cpu"

print(f"使用设备: {device} ({platform.system()} {platform.machine()})")

if device == "cpu":
    torch.set_num_threads(cpu_count())
    torch.set_num_interop_threads(cpu_count())

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

CACHE_DIR = "translation_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
BATCH_SIZE = 32 if device == "mps" else 64  # MPS需要更小的batch size
THRESHOLD = 0.82  # 语义相似度阈值


# -------------------------------------------

# ------------------- 模型加载 -------------------
def load_model():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)

    # 特殊处理MPS设备
    model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        device=device
    )

    # MPS设备需要关闭某些优化
    if device == "mps":
        model._target_device = torch.device("mps")
        for module in model._modules.values():
            if hasattr(module, "to"):
                module.to("mps")

    return model


model = load_model()
print("模型加载成功")


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
        self.df.to_csv(self.cache_file, index=False)


# ------------------- 文本匹配 -------------------
class TextMatcher:
    def __init__(self):
        self.cache = {}

    @staticmethod
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        return str(text).strip()

    def encode_texts(self, texts):
        """分批编码文本，兼容 CPU/GPU"""
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_embeddings = model.encode(batch_texts, batch_size=BATCH_SIZE, device=device, convert_to_tensor=True)
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)

    def compute_similarity(self, source_embedding, target_embeddings):
        return cosine_similarity(source_embedding.unsqueeze(0), target_embeddings).flatten()


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


# ------------------- 主逻辑 -------------------
def main():
    translation_cache = TranslationCache()
    matcher = TextMatcher()
    psv_df, ons_df, pymo_df = load_files()

    # 预处理目标文本
    ons_texts = [matcher.preprocess_text(t) for t in ons_df['zh_text'].tolist() if matcher.preprocess_text(t)]
    pymo_texts = [matcher.preprocess_text(t) for t in pymo_df['zh_text'].tolist() if matcher.preprocess_text(t)]

    # 预先编码目标文本
    print("编码 ONS 文本向量...")
    ons_embeddings = matcher.encode_texts(ons_texts) if ons_texts else None
    print("编码 PYMO 文本向量...")
    pymo_embeddings = matcher.encode_texts(pymo_texts) if pymo_texts else None

    results = []
    pbar = tqdm(total=len(psv_df), desc="处理进度")

    for _, row in psv_df.iterrows():
        ja_text = matcher.preprocess_text(row['ja_text'])
        best_match = None
        best_source = None
        best_sim = 0

        # 编码当前日文
        source_embedding = matcher.encode_texts([ja_text])[0]

        # 1. ONS匹配
        if ons_embeddings is not None:
            sims = matcher.compute_similarity(source_embedding, ons_embeddings)
            max_idx = sims.argmax()
            max_sim = sims[max_idx].item()
            if max_sim >= THRESHOLD:
                best_match = ons_texts[max_idx]
                best_source = "ons"
                best_sim = max_sim

        # 2. PYMO匹配
        if pymo_embeddings is not None and best_sim < 0.93:
            sims = matcher.compute_similarity(source_embedding, pymo_embeddings)
            max_idx = sims.argmax()
            max_sim = sims[max_idx].item()
            if max_sim >= THRESHOLD and max_sim > best_sim:
                best_match = pymo_texts[max_idx]
                best_source = "pymo"
                best_sim = max_sim

        # 3. 翻译缓存
        if not best_match:
            cached_trans = translation_cache.get_translation(ja_text)
            if cached_trans:
                best_match = cached_trans
                best_source = "translate(cached)"
            else:
                zh_text = f"[翻译]{ja_text}"  # 占位
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

    # 保存结果
    pd.DataFrame(results).to_csv('final_translations.csv', index=False, encoding='utf-8')
    print("处理完成，结果已保存到 final_translations.csv")


if __name__ == "__main__":
    main()

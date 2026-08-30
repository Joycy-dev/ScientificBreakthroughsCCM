import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import re
import time

# ============================================================== 
# 配置
# ============================================================== 
EMB_MODE = "cls"   # "cls" or "mask"
BATCH_SIZE = 32     # 批推理大小，可根据显存调整
MAX_LENGTH = 512
MODEL_NAME = "allenai/scibert_scivocab_uncased"
USE_FP16 = True     # 在 GPU 上尝试使用 mixed precision / half 模式（若不稳定可设为 False）

# ============================================================== 
# 1. 稳健 CSV 读取器
# ============================================================== 
def robust_read_csv(path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    for enc in encodings:
        try:
            print(f"尝试使用 {enc} 编码读取：{path}")
            df = pd.read_csv(path, encoding=enc, dtype=str, on_bad_lines="skip")
            print(f"✔ 成功使用 {enc} 读取：{path}")
            return df
        except Exception as e:
            print(f"✘ 使用 {enc} 失败：{e}")
    raise ValueError(f"无法读取 CSV：{path}")

# ============================================================== 
# 2. 加载 SciBERT（尽量一次加载并支持多 GPU）
# ============================================================== 
def load_scibert():
    print("🔍 加载 SciBERT 模型 …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # 优化 cudnn
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        try:
            if USE_FP16:
                model.half()
                print("⚡ 已将模型转换为 FP16（half），请确保模型与 FP16 兼容")
        except Exception as e:
            print(f"⚠ 无法将模型转换为 FP16（继续使用 FP32）：{e}")

        if num_gpus > 1:
            print(f"♻ 检测到 {num_gpus} GPU，使用 DataParallel 加速")
            model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device("cpu")
    model.eval()
    print(f"✔ SciBERT 已加载 (device = {device})")
    return tokenizer, model, device

# ============================================================== 
# 3. Pooling 方法
# ============================================================== 
def mean_pooling(hidden_state, attention_mask):
    """
    attention_mask 加权平均 pooling，忽略 padding。
    hidden_state: (batch, seq_len, hidden_dim)
    attention_mask: (batch, seq_len)
    返回: (batch, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    masked_hidden = hidden_state * mask
    sum_hidden = masked_hidden.sum(dim=1)
    sum_mask = mask.sum(dim=1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    pooled = sum_hidden / sum_mask
    return pooled

# ============================================================== 
# 4. 批量编码文本（使用 pin_memory + non_blocking，以及 autocast）
# ============================================================== 
def encode_texts_batch(texts, tokenizer, model, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, use_cls_fallback=False):
    """
    texts: list[str]
    返回: numpy array, shape (len(texts), hidden_dim)
    说明:
      - 使用 tokenizer 的动态 padding（return_tensors='pt'）
      - 在 GPU 时对输出 tensors 使用 pin_memory -> to(device, non_blocking=True)
      - 在 GPU 时使用 autocast 进行混合精度推理
      - 对空文本可选择用 CLS 向量作为 fallback
    """
    all_embs = []
    n = len(texts)
    device_is_cuda = (device.type == "cuda")

    for start in tqdm(range(0, n, batch_size), desc="Batch encode", leave=False):
        batch_texts = texts[start:start+batch_size]
        is_empty = [True if (t is None or str(t).strip() == "") else False for t in batch_texts]
        prepared = [t if not (t is None or str(t).strip() == "") else "." for t in batch_texts]

        encoded = tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt"
        )

        # 如果使用 CUDA，则先 pin_memory，然后非阻塞搬运至 GPU
        if device_is_cuda:
            encoded = {k: v.pin_memory() for k, v in encoded.items()}
            encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
        else:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        # 推理（混合精度 + 无梯度）
        with torch.no_grad():
            use_autocast = device_is_cuda and USE_FP16
            if use_autocast:
                autocast_ctx = torch.cuda.amp.autocast()
            else:
                # Dummy context manager
                class _DummyCtx:
                    def __enter__(self): return None
                    def __exit__(self, exc_type, exc, tb): return False
                autocast_ctx = _DummyCtx()

            with autocast_ctx:
                output = model(**encoded)
                # 有些模型返回 tuple，第一个元素含 last_hidden_state
                if hasattr(output, "last_hidden_state"):
                    last_hidden = output.last_hidden_state
                else:
                    # 兼容：输出可能是 tuple
                    last_hidden = output[0]

                pooled = mean_pooling(last_hidden, encoded["attention_mask"])  # (batch, hidden_dim)
                cls_vec = last_hidden[:, 0, :]  # (batch, hidden_dim)

                if use_cls_fallback:
                    batch_emb = []
                    for i, empty in enumerate(is_empty):
                        if empty:
                            emb = cls_vec[i]
                        else:
                            emb = pooled[i]
                        batch_emb.append(emb.detach().cpu().float().numpy())
                    batch_emb = np.stack(batch_emb, axis=0)
                else:
                    batch_emb = pooled.detach().cpu().float().numpy()

        all_embs.append(batch_emb)

    # 合并
    if all_embs:
        all_embs = np.vstack(all_embs)
    else:
        hidden_size = model.module.config.hidden_size if hasattr(model, "module") else model.config.hidden_size
        all_embs = np.zeros((0, hidden_size), dtype=np.float32)

    all_embs = all_embs.astype(np.float32)
    return all_embs

# ============================================================== 
# 5A. encode_dataframe 返回 (df_meta, embeddings) 而不是把 emb 列展开到 DataFrame（节省 IO）
# ============================================================== 
def encode_dataframe_cls(df, tokenizer, model, device):
    df = df.copy()
    if "title" not in df.columns:
        raise ValueError("DataFrame 中缺少 'title' 列")
    if "abstract" not in df.columns:
        df["abstract"] = ""

    df["title"] = df["title"].astype(str).fillna("")
    df["abstract"] = df["abstract"].astype(str).fillna("")

    titles = df["title"].tolist()
    abstracts = df["abstract"].tolist()

    print("🔁 批量编码 title（CLS 方案）...")
    v_titles = encode_texts_batch(titles, tokenizer, model, device, use_cls_fallback=True)
    print("🔁 批量编码 abstract（CLS 方案）...")
    v_abstracts = encode_texts_batch(abstracts, tokenizer, model, device, use_cls_fallback=True)

    vectors = np.concatenate([v_titles, v_abstracts], axis=1)  # (n, hidden*2)
    return df, vectors

# ============================================================== 
# 5B. attention-mask 版本：masked mean pooling，空摘要复用标题向量
# ============================================================== 
def encode_dataframe_mask(df, tokenizer, model, device):
    df = df.copy()
    if "title" not in df.columns:
        raise ValueError("DataFrame 中缺少 'title' 列")
    if "abstract" not in df.columns:
        df["abstract"] = ""

    df["title"] = df["title"].astype(str).fillna("")
    df["abstract"] = df["abstract"].astype(str).fillna("")

    titles = df["title"].tolist()
    abstracts = df["abstract"].tolist()

    print("🔁 批量编码 title（mask 方案）...")
    v_titles = encode_texts_batch(titles, tokenizer, model, device, use_cls_fallback=False)
    print("🔁 批量编码 abstract（mask 方案）...")
    v_abstracts = encode_texts_batch(abstracts, tokenizer, model, device, use_cls_fallback=False)

    # 对于摘要为空的，使用标题向量代替摘要槽位（保持原逻辑）
    for i, a in enumerate(abstracts):
        if a is None or str(a).strip() == "":
            v_abstracts[i] = v_titles[i].copy()

    vectors = np.concatenate([v_titles, v_abstracts], axis=1)  # (n, hidden*2)
    return df, vectors

# ============================================================== 
# 5C. 统一入口：encode_dataframe -> 返回 (df_meta, vectors)
# ============================================================== 
def encode_dataframe(df, tokenizer, model, device):
    mode = EMB_MODE.lower()
    if mode == "cls":
        print("⚙ 使用 CLS fallback 方案进行批量编码 …")
        return encode_dataframe_cls(df, tokenizer, model, device)
    elif mode == "mask":
        print("⚙ 使用 attention-mask 方案进行批量编码 …")
        return encode_dataframe_mask(df, tokenizer, model, device)
    else:
        raise ValueError(f"未知的 EMB_MODE = {EMB_MODE}，请设置为 'cls' 或 'mask'")

# ============================================================== 
# 6. 计算异质性（使用 numpy 快速计算余弦相似度）
# ============================================================== 
def compute_heterogeneity_from_vectors(v_paper, v_refs, meta_paper_row, idx):
    """
    v_paper: (1, dim)
    v_refs: (n_refs, dim)
    使用 numpy 归一化 + 矩阵乘法替代 sklearn.cosine_similarity
    返回: 1 行 DataFrame
    """
    if v_refs.shape[0] == 0:
        print(f"⚠ idx {idx}: 参考文献为 0 篇，将异质性设为 NaN")
        return pd.DataFrame([{
            "idx": idx,
            "paper_id": meta_paper_row.get("paper_id", ""),
            "title": meta_paper_row.get("title", ""),
            "num_refs": 0,
            "mean_similarity": np.nan,
            "heterogeneity": np.nan
        }])

    # 防止 0 向量导致除零
    a = v_paper.astype(np.float32)
    b = v_refs.astype(np.float32)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.clip(a_norm, 1e-9, None)
    b_norm = np.clip(b_norm, 1e-9, None)

    a_unit = a / a_norm
    b_unit = b / b_norm

    sims = np.dot(a_unit, b_unit.T)[0]  # (n_refs,)
    mean_sim = float(np.mean(sims))    # 平均相似度
    max_sim = float(np.max(sims))      # 最大相似度
    min_sim = float(np.min(sims))      # 最小相似度
    std_sim = float(np.std(sims))      # 标准差
    hetero = max_sim - min_sim

    return pd.DataFrame([{
        "idx": idx,
        "paper_id": meta_paper_row.get("paper_id", ""),
        "title": meta_paper_row.get("title", ""),
        "num_refs": int(v_refs.shape[0]),
        "mean_similarity": mean_sim,
        "max_similarity": max_sim,      
        "min_similarity": min_sim,      
        "std_similarity": std_sim,      
        "heterogeneity": hetero
    }])

# ============================================================== 
# 7. 处理每个编号：现在不再在这里加载模型（model/tokenizer/device 由外部传入）
# ============================================================== 
def process_pair(paper_file, ref_file, output_folder, idx, tokenizer, model, device):
    print(f"\n==================== 开始处理编号 {idx} ====================\n")

    df_p = robust_read_csv(paper_file)
    df_r = robust_read_csv(ref_file)

    df_p.fillna("", inplace=True)
    df_r.fillna("", inplace=True)

    print(f"📘 批量编码目标论文 {idx} ...")
    df_p_meta, v_p = encode_dataframe(df_p, tokenizer, model, device)  # v_p: (n_p, dim)
    print(f"📗 批量编码参考文献 {idx} ...")
    df_r_meta, v_r = encode_dataframe(df_r, tokenizer, model, device)  # v_r: (n_r, dim)

    os.makedirs(output_folder, exist_ok=True)
    mode = EMB_MODE.lower()

    # 保存 embeddings 为压缩 npz（减少 IO，比 csv 快且体积小）
    emb_out = os.path.join(output_folder, f"{idx}_emb_{mode}.npz")
    np.savez_compressed(emb_out, paper_embeddings=v_p, ref_embeddings=v_r)
    print(f"📦 已保存 embeddings：{emb_out}")

    # 保存元数据（尝试 parquet，否则回退 csv）
    meta_out = os.path.join(output_folder, f"{idx}_meta_{mode}.parquet")
    try:
        df_p_copy = df_p_meta.copy()
        df_p_copy["_source"] = "paper"
        df_r_copy = df_r_meta.copy()
        df_r_copy["_source"] = "ref"
        df_meta_all = pd.concat([df_p_copy, df_r_copy], ignore_index=True)
        df_meta_all.to_parquet(meta_out, index=False)
        print(f"📄 已保存元数据（parquet）：{meta_out}")
    except Exception as e:
        meta_out_csv = os.path.join(output_folder, f"{idx}_meta_{mode}.csv")
        df_p_copy = df_p_meta.copy()
        df_p_copy["_source"] = "paper"
        df_r_copy = df_r_meta.copy()
        df_r_copy["_source"] = "ref"
        df_meta_all = pd.concat([df_p_copy, df_r_copy], ignore_index=True)
        df_meta_all.to_csv(meta_out_csv, index=False, encoding="utf-8-sig")
        print(f"⚠ 无法保存 parquet（{e}），改为保存 csv：{meta_out_csv}")

    # 计算异质性（假设目标只有一行；如多行请调整）
    if v_p.shape[0] >= 1:
        v_paper = v_p[0].reshape(1, -1)
    else:
        v_paper = np.zeros((1, v_p.shape[1] if v_p.shape[1:] else 1), dtype=np.float32)

    result_df = compute_heterogeneity_from_vectors(v_paper, v_r, df_p_meta.iloc[0] if len(df_p_meta) > 0 else pd.Series(), idx)

    # 清理显存（如果使用 GPU）
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return result_df

# ============================================================== 
# 8. 扫描文件夹 + 顺序处理 + 汇总（模型只加载一次）
# ============================================================== 
def process_folder(
    input_folder=r"D:\work\论文\Experiment\Semantics\papers_abstract",
    output_folder=r"D:\work\论文\Experiment\Semantics\output"
):
    files = os.listdir(input_folder)

    # 提取所有编号
    indices = sorted(set(
        re.match(r"(\d+)_", f).group(1)
        for f in files if re.match(r"(\d+)_", f)
    ))

    if not indices:
        print("❌ 找不到编号文件！例如：0_paper.csv / 0_ref.csv")
        return

    print("📂 发现编号：", indices)

    # 只加载一次模型与 tokenizer
    tokenizer, model, device = load_scibert()

    hetero_results = []

    for idx in indices:
        paper_file = os.path.join(input_folder, f"{idx}_paper.csv")
        ref_file = os.path.join(input_folder, f"{idx}_ref.csv")

        if os.path.exists(paper_file) and os.path.exists(ref_file):
            try:
                result_df = process_pair(paper_file, ref_file, output_folder, idx, tokenizer, model, device)
                hetero_results.append(result_df)
            except Exception as e:
                print(f"✘ 处理 {idx} 时出现错误：{e}")
        else:
            print(f"⚠ 跳过 {idx}（缺少 {idx}_paper.csv 或 {idx}_ref.csv）")

    # 汇总所有 idx 的异质性结果到一个总表（文件名根据 EMB_MODE 动态）
    if hetero_results:
        all_hetero = pd.concat(hetero_results, ignore_index=True)
        mode = EMB_MODE.lower()
        semantics_dir = os.path.dirname(input_folder)
        hetero_out = os.path.join(semantics_dir, f"heterogeneity_output614_{mode}.csv")
        all_hetero.to_csv(hetero_out, index=False, encoding="utf-8-sig")
        print(f"\n🎉 已生成总异质性表：{hetero_out}")
    else:
        print("⚠ 没有任何异质性结果可以汇总。")

# ============================================================== 
# 主入口
# ============================================================== 
if __name__ == "__main__":
    process_folder()
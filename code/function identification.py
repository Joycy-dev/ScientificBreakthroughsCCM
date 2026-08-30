import os
import io
import csv
import math
import base64
from typing import List, Dict, Tuple
import json
import requests
import re
import time
import concurrent.futures
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer.layout").setLevel(logging.ERROR)

# 本地 GPU 推理相关导入（可选）
import threading
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

# 依赖：pip install openai
from openai import OpenAI

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# ================================
# 路径设置
# ================================
WORKSPACE_ROOT = r"D:\work\论文\Experiment\function"
PAPERS_DIR = os.path.join(WORKSPACE_ROOT, "paper")

# 日志/记录文件
LOG_LOCK = threading.Lock()
BAD_PDF_LOG = os.path.join(WORKSPACE_ROOT, "bad_pdfs.txt")
ERROR_CSV = os.path.join(WORKSPACE_ROOT, "processing_errors.csv")

# ================================
# 引文抽取 + 引文功能识别 Prompt
# ================================
USER_PROMPT_INSTRUCTION = """You are an expert in scientific text analysis. Your task is to extract all the complete sentences containing reference-style from the body of the paper according to the semantics of the article, and at the same time classify the citation sentences into shallow citations, moderate citations, and deep citations according to the context semantics, and annotate them to achieve zero omissions and zero truncations.
Strictly observe all of the following rules:
1. Extraction rules
(1) You must scan paragraph by paragraph and sentence by sentence, and you must not skip any part. 
(2) You need to identify all citation scenarios, including but not limited to:
- Citations in body paragraphs
- Embed references in algorithms, training steps, pseudocode, experiment settings
- References in the Table/Figure title or caption
- List items, comments, citations in spread text
- There is no limit to the citation position (beginning, middle, end, parentheses, line breaks, etc. must be extracted).
(3) You must extract all reference-styles, including but not limited to:
- (Author Year)、(Author et al. Year)、(A Year; B Year; C Year)、（the method (Author Year) shows… ）
- The author's name appears in the sentence but not in parentheses
- References in list or description items
- Duplicate, identical quotes that appear in different places in the article, all retained
2. Identify the rules
(1) Classification definition
- Deep citations: Citations are used for research breakthroughs, such as proposing new theories, models, or improving cited literature work based on cited literature. 
- Moderate citations: Citations are used for research implementation, such as detailed descriptions or employing of cited literature methods, algorithms, datasets, theoretical frameworks, evaluation indicators, etc. 
- Shallow Citation: Citations are only used for background introductions, literature lists, theoretical explanations, or narrative supplements, and do not participate in the research process. 
(2) Judgment process
- Breakthrough Detection: Citations are classified as deep citations if they are used as a key source for theoretical expansions, method improvements, framework enhancements, or new methodological theories proposed.
- Application Detection: Moderate citations if they are described in detail or used to study methods, algorithm calls, experimental operations, data sources, or compare and contrast. Note: Even if the sentence only describes the method and does not explicitly appear with words of type "use", it should be classified as moderately citated as long as the content is part of the research method chain (such as algorithm step descriptions, framework composition, or task execution flow).
- Background Detection: If a citation is only used for examples, roundups, definitions, or background notes, it is classified as a shallow citation.
(3) Decision-making priority
Deep citations >Moderate citations > Shallow citations
3. Classification examples
(1) Deep Citation
For example:
-To overcome these difficulties, we propose a new white-box DNN watermarking scheme based on multi-task learning (MTL) (Sener and Koltun 2018), MTLSign, as shown in Fig. 1.
-We extend the training regime proposed in (Hafner et al. 2019a) to include the training and application of ABPS.
-Compared to the prior works (Bach and Levy 2019; Ene, Nguyen, and Vladu 2021), our algorithms set the step sizes based on the operator value differences instead of the iterate movement.
(2) Moderate Citation
For example: 
-In this work, we use syntactically co-safe Linear Temporal Logic (scLTL) (Kupferman and Vardi 2001) as our specification language.
-Following (Nesterov 2007), we choose an arbitrary point x ∈ X.
-In our experiments, we focused on two natural domains: the "rocket" domain (Blum and Furst 1995) and the "logistics" domain (Veloso 1992).
(3) Shallow Citation
For example:
-Watermarking is an influential method for DNN IP protection (Uchida et al. 2017).
-To rectify the first issue, several works have been developed to design adaptive algorithms that automatically adapt to the smoothness of the problem (Bach and Levy 2019; Ene, Nguyen, and Vladu 2021).
-In this work, we focus on a particular family of approaches known as shielding (Alshiekh et al. 2018; Anderson et al. 2020; Giacobbe et al. 2021; ElSayed-Aly et al. 2021; Pranger et al. 2021).
4. Output requirements
- Only output complete sentences with citations
- Maintain the original order of each sentence
- No sentences may be rewritten, supplemented, or shortened
- If there is a line split, it needs to be merged before the full version of the sentence is output
- Add tags to the category output: [Deep citation], [Moderate citation], [Shallow Citation]
- Each sentence takes a separate line
- Do not output any redundant content (e.g. preface, summary, explanation, sentences without citations, etc.)
"""

# ================================
# 本地 GPU 模型缓存与辅助函数（可选使用，需安装 transformers + torch 并确保 GPU 可用）
# ================================
LOCAL_GEN = None
LOCAL_LOCK = threading.Lock()
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "gpt2-medium")  # 可通过环境变量调整模型名

def init_local_model():
    global LOCAL_GEN
    if LOCAL_GEN is not None:
        return LOCAL_GEN
    if pipeline is None or torch is None or not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        raise RuntimeError("本地 GPU 模型不可用，请安装 transformers/torch 并确保 GPU 可用")
    try:
        # 加载 pipeline 到 GPU device 0；对于大型模型请通过 LOCAL_MODEL_NAME 指定
        LOCAL_GEN = pipeline("text-generation", model=LOCAL_MODEL_NAME, device=0)
    except Exception as e:
        LOCAL_GEN = None
        raise RuntimeError(f"加载本地模型失败: {e}")
    return LOCAL_GEN

def local_generate(prompt: str) -> str:
    init_local_model()
    with LOCAL_LOCK:
        out = LOCAL_GEN(prompt, max_length=2048, do_sample=False, return_full_text=False)
    if isinstance(out, list) and len(out) > 0:
        first = out[0]
        if isinstance(first, dict):
            if "generated_text" in first:
                return first["generated_text"]
            if "text" in first:
                return first["text"]
        return str(first)
    return str(out)

# ================================
# DeepSeek API
# ================================
def make_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client

# ================================
# 文本读取（PDF、Docx、Txt）
# ================================
def _merge_columns_for_page(page) -> str:
    try:
        w = getattr(page, "width", None)
        h = getattr(page, "height", None)
        if not w or not h:
            return page.extract_text() or ""

        # 将页面宽度分割为三列
        third = w / 3
        left = page.crop((0, 0, third, h)).extract_text() or ""
        middle = page.crop((third, 0, 2 * third, h)).extract_text() or ""
        right = page.crop((2 * third, 0, w, h)).extract_text() or ""

        # 判断是否为三列排版
        if left.strip() and middle.strip() and right.strip():
            # 如果三列都有内容，认为是三列排版，合并三列文本
            return left.rstrip() + "\n\n" + middle.rstrip() + "\n\n" + right.lstrip()

        # 判断为双列排版：如果左右列有内容，且中列为空
        mid = w / 2
        left = page.crop((0, 0, mid, h)).extract_text() or ""
        right = page.crop((mid, 0, w, h)).extract_text() or ""

        if not left.strip() and not right.strip():
            return page.extract_text() or ""  # 单列

        # 双列排版，合并左右两列文本
        return left.rstrip() + "\n\n" + right.lstrip()

    except Exception:
        return page.extract_text() or ""


def extract_text_from_pdf(path: str) -> str:
    if pdfplumber is None:
        raise RuntimeError("缺少 pdfplumber")
    texts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            texts.append(_merge_columns_for_page(p))
    return "\n\n".join(texts)

def extract_text_from_docx(path: str) -> str:
    if docx is None:
        raise RuntimeError("缺少 python-docx")
    d = docx.Document(path)
    return "\n\n".join(p.text for p in d.paragraphs)

def extract_text_from_txt(path: str) -> str:
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in [".doc", ".docx"]:
        return extract_text_from_docx(path)
    return extract_text_from_txt(path)

# ================================
# 文本分句 + 分块
# ================================
def _normalize_newlines_and_merge_broken_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    placeholder = "<PARA>"
    text = re.sub(r"\n{2,}", placeholder, text)
    text = re.sub(r"\n+", " ", text)
    return text.replace(placeholder, "\n\n")

def split_into_sentences(text: str) -> List[str]:
    text = _normalize_newlines_and_merge_broken_lines(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    sent_re = re.compile(r"(?<=[。！？；;.!?])\s+")
    sentences = []

    for para in paragraphs:
        if len(para) < 80:
            sentences.append(para)
            continue
        parts = sent_re.split(para)
        for p in parts:
            if p.strip():
                sentences.append(p.strip())
    return sentences

def create_chunks_from_text(text: str, chunk_size=12000, overlap_sentences=3):
    sents = split_into_sentences(text)
    if not sents:
        return []
    chunks = []
    cur = []
    cur_len = 0
    i = 0
    while i < len(sents):
        s = sents[i]
        if len(s) >= chunk_size:
            if cur:
                chunks.append(" ".join(cur))
                cur = []
                cur_len = 0
            chunks.append(s)
            i += 1
            continue
        if cur_len + len(s) + 1 <= chunk_size or not cur:
            cur.append(s)
            cur_len += len(s) + 1
            i += 1
        else:
            chunks.append(" ".join(cur))
            overlap = cur[-overlap_sentences:] if overlap_sentences > 0 else []
            cur = overlap.copy()
            cur_len = sum(len(x) + 1 for x in cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ================================
# 调用 DeepSeek：并行处理单文件（支持本地 GPU 推理）
# ================================
def call_deepseek_for_file(file_path: str) -> Tuple[str, str, str]:
    relname = os.path.relpath(file_path, PAPERS_DIR)

    # 预检查：如果是 PDF，尝试快速打开以判断是否损坏
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf" and pdfplumber is not None:
            try:
                with pdfplumber.open(file_path) as _pdf:
                    _ = len(_pdf.pages)
            except Exception as e:
                with LOG_LOCK:
                    with io.open(BAD_PDF_LOG, "a", encoding="utf-8") as bf:
                        bf.write(f"{file_path}\t{repr(e)}\n")
                return (file_path, "", f"PDF 打开失败或损坏：{e}")
    except Exception:
        pass

    try:
        text = extract_text(file_path)
    except Exception as e:
        return (file_path, "", f"文件解析失败：{e}")

    if not text.strip():
        return (file_path, "", "文件内容为空")

    chunks = create_chunks_from_text(text, 12000, 3)

    use_local_gpu = os.environ.get("USE_LOCAL_GPU_MODEL", "1") == "1" and (pipeline is not None) and (torch is not None) and (hasattr(torch, "cuda") and torch.cuda.is_available())

    if not use_local_gpu:
        try:
            client = make_client()
        except Exception as e:
            return (file_path, "", f"创建客户端失败：{e}")
    else:
        try:
            init_local_model()
        except Exception as e:
            return (file_path, "", f"本地 GPU 模型初始化失败：{e}")

    outputs = []
    for idx, chunk in enumerate(chunks):
        user_msg = f"文件: {relname}\n部分 {idx+1}/{len(chunks)}:\n\n{chunk}"
        try:
            if use_local_gpu:
                prompt = USER_PROMPT_INSTRUCTION + "\n\n" + user_msg
                text_out = local_generate(prompt)
            else:
                resp = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": USER_PROMPT_INSTRUCTION},
                        {"role": "user", "content": user_msg},
                    ],
                    stream=False
                )
                text_out = resp.choices[0].message.content
        except Exception as e:
            return (file_path, "", f"API调用/本地推理失败：{e}")

        outputs.append(text_out.strip())

    return (file_path, "\n".join(outputs), "")

# ================================
# ⭐ 主函数：处理全部论文 + 输出（并发策略根据是否使用本地 GPU 调整）
# ================================
def process_all_papers():
    if not os.path.isdir(PAPERS_DIR):
        raise RuntimeError(f"papers 目录不存在：{PAPERS_DIR}")

    SINGLE_OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "citation_output")
    os.makedirs(SINGLE_OUTPUT_DIR, exist_ok=True)

    SUMMARY_CSV = os.path.join(WORKSPACE_ROOT, "relevance_output.csv")

    # 收集文件
    file_list = []
    for root, _, files in os.walk(PAPERS_DIR):
        for fn in files:
            if not fn.startswith("."):
                file_list.append(os.path.join(root, fn))
    file_list = sorted(file_list)

    cpu = os.cpu_count() or 4

    use_local_gpu = os.environ.get("USE_LOCAL_GPU_MODEL", "1") == "1" and (pipeline is not None) and (torch is not None) and (hasattr(torch, "cuda") and torch.cuda.is_available())

    if use_local_gpu:
        max_workers = min(2, max(1, cpu))
        print("检测到本地 GPU 推理模式，已将并发线程数限制为：", max_workers)
    else:
        max_workers = min(8, max(2, cpu * 2))

    print(f"共找到 {len(file_list)} 篇论文，开始处理...\n")

    start_time = time.time()

    # 事先打开并写入汇总 CSV 的表头，后续每处理完一篇就追加一行
    summary_fieldnames = [
        "idx", "title", "num_all",
        "num_d", "num_m", "num_l",
        "P_d", "P_m", "P_l"
    ]
    summary_file = io.open(SUMMARY_CSV, "w", encoding="utf-8-sig", newline="")
    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fieldnames)
    summary_writer.writeheader()
    summary_file.flush()

    # 处理错误日志 CSV（记录每个发生错误的文件）
    err_file = io.open(ERROR_CSV, "w", encoding="utf-8-sig", newline="")
    err_writer = csv.DictWriter(err_file, fieldnames=["idx", "file_path", "error"])
    err_writer.writeheader()
    err_file.flush()

    label_re = re.compile(r"^\s*\[(深度引用|中度引用|浅度引用)\]\s*(.*)")

    total_papers = len(file_list)
    processed_count = 0

    print(f"开始并行调用模型（max_workers={max_workers}）...\n")

    # 并行调用 DeepSeek，并在每个 future 完成时即时保存结果
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_file = {ex.submit(call_deepseek_for_file, fp): fp for fp in file_list}
        for fut in concurrent.futures.as_completed(future_to_file):
            fp = future_to_file[fut]
            try:
                file_path, model_out, err = fut.result()
            except Exception as e:
                file_path = fp
                model_out = ""
                err = f"线程异常：{e}"

            # 增加计数（用于命名单篇 CSV）
            idx = processed_count
            processed_count += 1

            filename = os.path.basename(file_path)
            title, _ = os.path.splitext(filename)

            if err:
                # 记录错误 CSV
                err_writer.writerow({"idx": idx, "file_path": file_path, "error": err})
                err_file.flush()

                # 仍然创建单篇 CSV（仅 header），以便后续查看
                paper_name = f"paper_{idx}.csv"
                paper_path = os.path.join(SINGLE_OUTPUT_DIR, paper_name)
                with io.open(paper_path, "w", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["title", "sentence", "category"])
                    w.writeheader()

                # 在汇总中写入 0 的统计
                summary_writer.writerow({
                    "idx": idx,
                    "title": title,
                    "num_all": 0,
                    "num_d": 0,
                    "num_m": 0,
                    "num_l": 0,
                    "P_d": 0,
                    "P_m": 0,
                    "P_l": 0
                })
                summary_file.flush()

                print(f"[跳过] {file_path}: {err}")
                continue

            # 正常解析模型输出并写入单篇 CSV 与汇总
            parsed_rows = []
            num_d = num_m = num_l = 0

            for line in model_out.splitlines():
                s = line.strip()
                if not s:
                    continue
                m = label_re.match(s)
                if not m:
                    continue

                category = m.group(1)
                sentence = m.group(2).strip()

                if category == "深度引用":
                    num_d += 1
                elif category == "中度引用":
                    num_m += 1
                elif category == "浅度引用":
                    num_l += 1

                parsed_rows.append({
                    "title": title,
                    "sentence": sentence,
                    "category": category
                })

            num_all = num_d + num_m + num_l

            if num_all == 0:
                P_d = P_m = P_l = 0
            else:
                P_d = num_d / num_all
                P_m = num_m / num_all
                P_l = num_l / num_all

            # 保存单篇 CSV（即时）
            paper_name = f"paper_{idx}.csv"
            paper_path = os.path.join(SINGLE_OUTPUT_DIR, paper_name)

            with io.open(paper_path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["title", "sentence", "category"])
                w.writeheader()
                for r in parsed_rows:
                    w.writerow(r)

            # 将该篇统计写入汇总（即时追加）
            summary_writer.writerow({
                "idx": idx,
                "title": title,
                "num_all": num_all,
                "num_d": num_d,
                "num_m": num_m,
                "num_l": num_l,
                "P_d": round(P_d, 4),
                "P_m": round(P_m, 4),
                "P_l": round(P_l, 4),
            })
            summary_file.flush()

            # === 动态进度条（基于已处理数量）===
            done = processed_count
            bar_len = 25
            progress = done / total_papers if total_papers else 1.0
            filled = int(bar_len * progress)
            if filled >= bar_len:
                bar = "[" + "=" * bar_len + "]"
            else:
                bar = "[" + "=" * filled + ">" + "-" * (bar_len - filled - 1) + "]"

            elapsed = time.time() - start_time
            avg = elapsed / done
            eta = avg * (total_papers - done)

            print(
                f"{bar} {done}/{total_papers} 已完成 {paper_name} (引用句数={num_all})  "
                f"耗时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s",
                flush=True
            )

    # 关闭 summary 与 error 文件句柄
    summary_file.close()
    err_file.close()

    print("\n=== 全部完成 ===")
    print(f"单篇文件保存在：{SINGLE_OUTPUT_DIR}")
    print(f"汇总结果保存在：{SUMMARY_CSV}")
    print(f"处理错误记录：{ERROR_CSV}")
    print(f"损坏 PDF 列表：{BAD_PDF_LOG}")
    print(f"总耗时：{time.time() - start_time:.1f} 秒")


if __name__ == "__main__":
    process_all_papers()
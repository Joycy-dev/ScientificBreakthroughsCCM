import os
import time
import random
import requests
import pandas as pd


# =========================================================
# 全局 Session（支持可选 API Key）
# =========================================================
SESSION = None


def get_session():
    """初始化 requests.Session，自动附加 API Key（如果设置了）"""
    global SESSION
    if SESSION is None:
        SESSION = requests.Session()
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            print("🔑 已检测到 API Key，将使用授权访问。")
            SESSION.headers.update({"x-api-key": api_key})
        else:
            print("⚠ 未检测到 API Key，将以匿名方式访问（更容易触发 429）。")
    return SESSION


# =========================================================
# 带重试的请求工具（解决 429 / 5xx）
# =========================================================
def request_with_retry(url: str, desc: str, max_retries=3):
    session = get_session()
    last_resp = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=30)
            last_resp = resp
        except Exception as e:
            print(f"⚠ [{desc}] 请求异常，第 {attempt} 次：{e}")
            time.sleep(2 ** attempt)
            continue

        # 限流 429
        if resp.status_code == 429:
            wait = resp.headers.get("Retry-After", None)
            wait_seconds = int(wait) if wait and wait.isdigit() else 10
            print(f"⚠ [{desc}] HTTP 429 第 {attempt} 次，等待 {wait_seconds} 秒重试...")
            time.sleep(wait_seconds)
            continue

        # 服务器错误 5xx
        if 500 <= resp.status_code < 600:
            wait_seconds = 2 ** attempt
            print(f"⚠ [{desc}] HTTP {resp.status_code} 第 {attempt} 次，等待 {wait_seconds} 秒...")
            time.sleep(wait_seconds)
            continue

        return resp

    # 多次失败后返回最后一次响应
    return last_resp


# =========================================================
# 1. 搜索 title → paperId
# =========================================================
def search_paper_id(title: str):
    url = f'https://api.semanticscholar.org/graph/v1/paper/search/match?query={title}&fields=abstract,referenceCount,citationCount,influentialCitationCount,fieldsOfStudy,references.abstract,references.title,references.fieldsOfStudy'
   

    resp = request_with_retry(url, desc="Search")
    if resp is None:
        return None

    print(f"[Search] HTTP {resp.status_code} | {title[:60]}")

    if resp.status_code != 200:
        return None

    data = resp.json().get("data", [])
    if not data:
        return None

    return data[0].get("paperId")


# =========================================================
# 2. 通过 paperId 获取详细信息（含 references）
# =========================================================
def fetch_paper_detail(paper_id: str):
    fields = (
        "title,abstract,referenceCount,citationCount,influentialCitationCount,"
        "fieldsOfStudy,references.paperId,references.title,"
        "references.abstract,references.fieldsOfStudy"
    )

    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"

    resp = request_with_retry(url, desc="Detail")
    if resp is None:
        return None

    print(f"[Detail] HTTP {resp.status_code} | paperId={paper_id}")

    if resp.status_code != 200:
        return None

    return resp.json()


# =========================================================
# 3. 处理单篇论文（封装全文逻辑）
# =========================================================
def get_request(title: str):
    # 第一步：标题搜索
    paper_id = search_paper_id(title)
    if not paper_id:
        return {"state": 0, "msg": "No paperId", "title": title}

    # 第二步：获取完整信息
    detail = fetch_paper_detail(paper_id)
    if not detail:
        return {"state": 0, "msg": "Detail Failed", "title": title}

    # 主文献信息
    paper_row = [
        paper_id,
        title,
        detail.get("abstract", ""),
        detail.get("referenceCount", ""),
        detail.get("citationCount", ""),
        detail.get("influentialCitationCount", ""),
        detail.get("fieldsOfStudy", [""])[0] if detail.get("fieldsOfStudy") else "",
    ]

    # 参考文献信息
    ref_rows = []
    references = detail.get("references") or []

    for ref in references:
        ref_rows.append(
            [
                ref.get("paperId", ""),
                ref.get("title", ""),
                ref.get("abstract", ""),
                ref.get("fieldsOfStudy", [""])[0] if ref.get("fieldsOfStudy") else "",
                paper_id,
                title,
            ]
        )

    return {"state": 1, "paper_row": paper_row, "ref_rows": ref_rows}


# =========================================================
# 4. 批量处理 Excel
# =========================================================
def fetch_all_from_excel(excel_path: str, save_dir: str = "output"):
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_excel(excel_path)
    if "title" not in df.columns:
        raise ValueError("Excel 文件必须包含列：title")

    titles = df["title"].tolist()
    print(f"\n📌 将处理 {len(titles)} 篇论文...\n")

    for i, title in enumerate(titles):
        print(f"▶ 正在处理 {i+1}/{len(titles)} ...")

        # 防止触发限流（你可以调小为 1~3 秒）
        time.sleep(random.uniform(1, 3))

        rst = get_request(title)

        if rst["state"] != 1:
            print("❌ 跳过：未获取到数据\n")
            continue

        # 保存主论文
        paper_df = pd.DataFrame(
            [rst["paper_row"]],
            columns=[
                "paper_id",
                "title",
                "abstract",
                "referenceCount",
                "citationCount",
                "influentialCitationCount",
                "fieldsOfStudy",
            ],
        )
        paper_df.to_csv(os.path.join(save_dir, f"{i}_paper.csv"), index=False)

        # 保存参考文献
        ref_df = pd.DataFrame(
            rst["ref_rows"],
            columns=[
                "paper_id",
                "title",
                "abstract",
                "fieldsOfStudy",
                "focus_paper_id",
                "focus_paper_title",
            ],
        )
        ref_df.to_csv(os.path.join(save_dir, f"{i}_ref.csv"), index=False)

        print(f"✓ 已保存：{save_dir}\\{i}_paper.csv, {save_dir}\\{i}_ref.csv\n")


# =========================================================
# 5. 主入口
# =========================================================
if __name__ == "__main__":
    excel_file = r"D:\work\论文\Experiment\get_ref\data.xlsx"
    output_dir = r"D:\work\论文\Experiment\get_ref\output"

    fetch_all_from_excel(excel_file, output_dir)

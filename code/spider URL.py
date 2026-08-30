# spider16_fixed.py — 在会话丢失时自动重建 driver 并重试
import time
import random
import urllib.parse
import re
from pathlib import Path
import pandas as pd
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import requests

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15"
]

def init_driver(headless=False, proxy=None):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    ua = random.choice(USER_AGENTS)
    opts.add_argument(f"user-agent={ua}")
    if proxy:
        opts.add_argument(f'--proxy-server={proxy}')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(30)
    return driver

def recreate_driver(old_driver, headless, proxy):
    try:
        if old_driver:
            try:
                old_driver.quit()
            except Exception:
                pass
    except Exception:
        pass
    return init_driver(headless=headless, proxy=proxy), WebDriverWait(init_driver(headless=headless, proxy=proxy), 20)

def extract_hrefs_from_html(html):
    tree = etree.HTML(html)
    anchors = tree.xpath('//a[@href]')
    hrefs = []
    seen = set()
    for a in anchors:
        href = a.get('href')
        if not href:
            continue
        href_orig = href
        href_no_q = href.split('?', 1)[0].lower()
        text = ''.join(a.itertext() or []).strip().lower()
        is_pdf = False
        if re.search(r'\.pdf(\b|$)', href_no_q):
            is_pdf = True
        elif 'pdf' in href_no_q:
            is_pdf = True
        elif re.search(r'/(download|downloads?|attachment|viewfile|article/download|getfile)/', href_no_q):
            is_pdf = True
        elif 'scholar.googleusercontent.com' in href_orig or 'cache:' in href_orig:
            is_pdf = True
        elif 'pdf' in text:
            is_pdf = True
        if is_pdf:
            full = urllib.parse.urljoin('https://scholar.google.com', href_orig)
            if full not in seen:
                seen.add(full)
                hrefs.append(full)
    return hrefs

def find_follow_links(html):
    base = 'https://scholar.google.com'
    tree = etree.HTML(html)
    anchors = tree.xpath('//a[@href]')
    out = []
    seen = set()
    for a in anchors:
        href = a.get('href')
        if not href:
            continue
        href_l = href.lower()
        text = ''.join(a.itertext() or '').strip().lower()
        take = False
        if 'cluster=' in href_l or 'cluster%3d' in href_l:
            take = True
        elif 'scholar.googleusercontent.com' in href_l or 'cache:' in href_l:
            take = True
        elif '所有' in text and '版本' in text:
            take = True
        elif 'all' in text and 'version' in text:
            take = True
        if take:
            full = urllib.parse.urljoin(base, href)
            if full not in seen:
                seen.add(full)
                out.append(full)
    return out

def _sanitize_filename(s):
    s2 = re.sub(r'[\\/*?:"<>|]', '_', s)
    return s2[:100]

def detect_captcha(driver=None, html=None):
    """
    启发式检测页面是否触发验证码/反爬（不尝试绕过）。
    检测项（任一满足则认为可能需要人工干预）：
    - HTML 中包含 'captcha'、'unusual traffic' 等关键词
    - 存在常见的 reCAPTCHA iframe
    - 页面 title 提示 “are you a robot” 等
    """
    if html is None and driver is not None:
        try:
            html = driver.page_source
        except Exception:
            html = ''
    if not html:
        return False
    h = html.lower()
    indicators = [
        'captcha', 'recaptcha', 'are you a robot', 'unusual traffic',
        'please show you are human', '为什么出现此问题', '访问过于频繁'
    ]
    for kw in indicators:
        if kw in h:
            return True
    # 进一步检查常见 iframe 标记
    if '<iframe' in h and ('recaptcha' in h or 'g-recaptcha' in h):
        return True
    return False

def wait_for_manual_solve(debug_dir, idx, title_str, driver):
    """
    当检测到验证码时：保存当前页面和截图，提示用户在浏览器中完成验证后按回车继续。
    """
    safe = _sanitize_filename(f"{idx}_{title_str}")
    html_file = debug_dir / f"{safe}_captcha.html"
    try:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        try:
            screenshot_file = debug_dir / f"{safe}_captcha.png"
            driver.save_screenshot(str(screenshot_file))
            print(f"[{idx}] 已保存 CAPTCHA 页面与截图: {html_file}, {screenshot_file}")
        except Exception:
            print(f"[{idx}] 已保存 CAPTCHA 页面: {html_file}（截图失败）")
    except Exception as e:
        print(f"[{idx}] 保存 CAPTCHA 调试页面失败: {e}")
    print(f"[{idx}] 检测到可能的人机验证，请在打开的浏览器中完成验证后按回车继续（或输入 skip 并回车跳过此条）。")
    choice = input().strip().lower()
    if choice == 'skip':
        return False
    # 等待用户按回车后再检查页面是否仍有 CAPTCHA 指示
    time.sleep(1)
    try:
        new_html = driver.page_source
        if detect_captcha(driver=None, html=new_html):
            print(f"[{idx}] 验证完成后系统仍检测到疑似验证码，请确认已完成人工验证，或输入 skip 跳过。再次按回车继续。")
            choice2 = input().strip().lower()
            if choice2 == 'skip':
                return False
    except Exception:
        pass
    return True

def safe_get_with_recreate(driver, wait, url, headless, proxy, retries=1):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            return driver, wait, driver.page_source
        except (WebDriverException, TimeoutException) as e:
            last_exc = e
            msg = str(e).lower()
            if 'invalid session id' in msg or 'session not created' in msg or 'chrome not reachable' in msg:
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = init_driver(headless=headless, proxy=proxy)
                wait = WebDriverWait(driver, 20)
                continue
            else:
                raise
    # if all retries failed, raise the last exception
    raise last_exc

def safe_page_source(driver, headless, proxy):
    try:
        return driver.page_source
    except WebDriverException as e:
        msg = str(e).lower()
        if 'invalid session id' in msg or 'chrome not reachable' in msg:
            try:
                driver.quit()
            except Exception:
                pass
            driver = init_driver(headless=headless, proxy=proxy)
            # caller is expected to navigate again if needed
            return ''
        else:
            return ''

def safe_screenshot(driver, path):
    try:
        driver.save_screenshot(str(path))
        return True
    except Exception:
        return False

def main(headless=False, proxy=None):
    base_dir = Path(__file__).parent
    excel_path = base_dir / 'physiology or medicine data.xlsx'
    if not excel_path.exists():
        print(f"找不到文件: {excel_path}")
        return

    df = pd.read_excel(excel_path, engine='openpyxl')
    if 'title' not in df.columns:
        print("Excel 中没有 title 列")
        return

    extracted = []
    output_path = excel_path.parent / (excel_path.stem + '_extracted.xlsx')
    debug_dir = base_dir / 'debug_snapshots2'
    debug_dir.mkdir(exist_ok=True)

    driver = init_driver(headless=headless, proxy=proxy)
    wait = WebDriverWait(driver, 20)

    try:
        for idx, title in enumerate(df['title'].fillna('').astype(str), 1):
            title_str = title.strip()
            if not title_str:
                extracted.append('')
                print(f"[{idx}] 空标题，跳过")
                continue

            q = urllib.parse.quote(title_str)
            url = f'https://scholar.google.com/scholar?q={q}'
            try:
                try:
                    driver, wait, html = safe_get_with_recreate(driver, wait, url, headless, proxy, retries=1)
                except Exception as e:
                    print(f"[{idx}] 无法加载页面（初次/重试均失败）：{e}")
                    extracted.append('')
                    continue

                time.sleep(random.uniform(3, 5.5))
                # 再次保证 html 不为空
                if not html:
                    try:
                        html = driver.page_source
                    except Exception:
                        html = ''

                if detect_captcha(driver=driver, html=html):
                    ok = wait_for_manual_solve(debug_dir, idx, title_str, driver)
                    if not ok:
                        print(f"[{idx}] 用户选择跳过或验证失败，记录为空并继续")
                        extracted.append('')
                        time.sleep(random.uniform(5, 8))
                        continue
                    try:
                        html = driver.page_source
                    except Exception:
                        html = ''

                hrefs = extract_hrefs_from_html(html)

                if not hrefs:
                    follow_links = find_follow_links(html)
                    if follow_links:
                        for fidx, follow in enumerate(follow_links, 1):
                            try:
                                try:
                                    driver, wait, html2 = safe_get_with_recreate(driver, wait, follow, headless, proxy, retries=1)
                                except Exception:
                                    continue
                                time.sleep(random.uniform(3.0, 7.0))
                                if detect_captcha(driver=driver, html=html2):
                                    ok = wait_for_manual_solve(debug_dir, idx, title_str, driver)
                                    if not ok:
                                        continue
                                    try:
                                        html2 = driver.page_source
                                    except Exception:
                                        html2 = ''
                                hrefs2 = extract_hrefs_from_html(html2)
                                if hrefs2:
                                    hrefs.extend(hrefs2)
                                    print(f"[{idx}] 从跟随链接找到 {len(hrefs2)} 个候选：{follow}")
                                    break
                            except Exception:
                                continue

                if not hrefs:
                    safe = _sanitize_filename(f"{idx}_{title_str}")
                    html_file = debug_dir / f"{safe}.html"
                    try:
                        with open(html_file, 'w', encoding='utf-8') as f:
                            f.write(html)
                        screenshot_file = debug_dir / f"{safe}.png"
                        if safe_screenshot(driver, screenshot_file):
                            print(f"[{idx}] 未找到 PDF，已保存 HTML 与截图: {html_file}, {screenshot_file}")
                        else:
                            print(f"[{idx}] 未找到 PDF，已保存 HTML: {html_file}（截图失败）")
                    except Exception as e:
                        print(f"[{idx}] 保存调试页面失败: {e}")
                else:
                    print(f"[{idx}] 标题: {title_str} -> 找到 {len(hrefs)} 个 PDF/候选链接（仅保留第 1 个）")

                first_href = hrefs[0] if hrefs else ''
                extracted.append(first_href)
            except Exception as e:
                print(f"[{idx}] 查询出错: {e}")
                try:
                    try:
                        html = driver.page_source if driver else ''
                    except Exception:
                        html = ''
                    safe = _sanitize_filename(f"{idx}_{title_str}")
                    err_html = debug_dir / f"{safe}_error.html"
                    with open(err_html, 'w', encoding='utf-8') as f:
                        f.write(html)
                    try:
                        screenshot_file = debug_dir / f"{safe}_error.png"
                        if safe_screenshot(driver, screenshot_file):
                            print(f"[{idx}] 已保存出错页面和截图: {err_html}, {screenshot_file}")
                        else:
                            print(f"[{idx}] 已保存出错页面: {err_html}（截图失败）")
                    except Exception:
                        print(f"[{idx}] 截图失败")
                except Exception as e2:
                    print(f"[{idx}] 保存出错调试信息失败: {e2}")
                extracted.append('')

            # 随机间隔以减小被封禁风险（保持人为节奏）
            time.sleep(random.uniform(8, 18.0))

    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass

    df['extracted_hrefs'] = extracted
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"完成，已将 extracted_hrefs 列写入: {output_path}")
    print(f"调试文件保存在: {debug_dir}（若有）")

def fetch_data(url, proxy=None):
    try:
        proxies = {'http': proxy, 'https': proxy} if proxy else None
        response = requests.get(url, proxies=proxies, timeout=5, headers={'User-Agent': random.choice(USER_AGENTS)})
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"请求错误：{e}")
        return None

if __name__ == '__main__':
    proxy = None
    main(headless=False, proxy=proxy)
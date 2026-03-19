import time
import os
import tempfile
import random
from tqdm import tqdm
from db import is_aldready_processed,save_pdf_record,get_db
from ingest import extract_text_from_pdf,chunk_text,setup_pinecone_index
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from curl_cffi import requests as cf_requests
import time,random
import json
# import browser_cookie3
load_dotenv()
BASE_URL   = "https://www.justice.gov/epstein/doj-disclosures"
NAMESPACE  = "epstein-docs"
DELAY      = 1.5    # seconds between requests — be polite to DOJ server
TIMEOUT    = 30     # seconds before giving up on a download
# HEADERS    = {
#     "User-Agent": (
#         "Mozilla/5.0 (compatible; GovTransparencyRAG/1.0; "
#         "educational research project)"
#     )
# }
HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":  "document",
    "Sec-Fetch-Mode":  "navigate",
    "Sec-Fetch-Site":  "none",
    "Sec-Fetch-User":  "?1",
}


DATASET_PAGES = [
    # "https://www.justice.gov/epstein/doj-disclosures/data-set-1-files",
    # "https://www.justice.gov/epstein/doj-disclosures/data-set-2-files",
    # "https://www.justice.gov/epstein/doj-disclosures/data-set-3-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-4-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-5-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-6-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-7-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-8-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-9-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-10-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-11-files",
    "https://www.justice.gov/epstein/doj-disclosures/data-set-12-files",
]
# def scrape_pdf_links(page_url:str)->list[dict]:
#     """
#     Scrape all PDF links from a DOJ dataset page.
#     Returns list of {filename, url, dataset} dicts.
#     """
#     full_page_url=page_url
#     print(f"\n🔍 Scraping: {full_page_url}")
#     try:
#         res=requests.get(full_page_url,headers=HEADERS,timeout=TIMEOUT)
#         res.raise_for_status()
#     except Exception as e:
#         print(f"  ✗ Failed to fetch page: {e}")
#         return []
#     soup=BeautifulSoup(res.text,"html.parser") #used to fetch href links exactly
#     links=[]
#     dataset=full_page_url.strip("/").split("/")[-1] #extracts the last part of url eg. dataset-1
        
#     for a in soup.find_all("a",href=True):
#         href=a["href"]
#         if href.lower().endswith(".pdf"):
#             full_url=urljoin("https://www.justice.gov",href)
#             filename=href.split("/")[-1]
#             links.append({
#                 "filename":filename,
#                 "url":full_url,
#                 "dataset":dataset
#             })
#     print(f"  ✓ Found {len(links)} PDF links")
#     time.sleep(DELAY)

def scrape_pdf_links(page_url: str) -> list[dict]:
    all_links=[]
    dataset=page_url.strip("/").split("/")[-1]
    #session exactly looks like chrome at TLS layer
    session=cf_requests.Session(impersonate="chrome120")
    print(f"\n🏠 Loading: {page_url}")
    res = session.get(page_url, timeout=TIMEOUT)
    res.raise_for_status()
    soup=BeautifulSoup(res.text,"html.parser")
    last_page=0
    for a in soup.find_all("a",href=True):
        if "?page=" in a["href"]:
            try:
                n = int(a["href"].split("?page=")[-1])
                last_page = max(last_page, n)
            except ValueError:
                pass
    print(f"  📋 {last_page + 1} pages (0–{last_page})")
    #page 0 
    for a in soup.find_all("a",href=True):
        if a["href"].lower().endswith(".pdf"):
            all_links.append({
                "filename":a["href"].split("/")[-1],
                "url":      urljoin("https://www.justice.gov", a["href"]),
                "dataset":dataset
            })
    print(f"  ✓ Page 0: {len(all_links)} PDFs")
    #page 1 to last page
    for i in range(1,last_page+1):
        time.sleep(random.uniform(2,4))
        full_url=f"{page_url}?page={i}"
        print(f"Scraping  🔍 {full_url}..")
        #different sessions for different pages
        res=session.get(
            full_url,
            headers={"Referer": page_url if i == 1 else f"{page_url}?page={i-1}"},
            timeout=TIMEOUT
        )
        res.raise_for_status()
        soup=BeautifulSoup(res.text,"html.parser")
        links = [
            {
                "filename": a["href"].split("/")[-1],
                "url":      urljoin("https://www.justice.gov", a["href"]),
                "dataset":  dataset,
            }
            for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith(".pdf")
        ]
        print(f"  ✓ Page {i}: {len(links)} PDFs")
        all_links.extend(links)
    print(f"\n📄 {dataset}: {len(all_links)} total PDFs")
    return all_links
# scrape_pdf_links("https://www.justice.gov/epstein/doj-disclosures/data-set-8-files")
# test passed successfully

# def download_pdf(url:str)->str | None:
#     """Download a PDF into a temp file, return path."""
#     try:
#         session=cf_requests.Session(impersonate="chrome120")
#         res=session.get(url,timeout=TIMEOUT,stream=True)
#         # setting session cookie for age restriction
#         session.cookies.set("justiceGate", "true", domain="www.justice.gov", path="/")
#         res.raise_for_status()
#         tmp=tempfile.NamedTemporaryFile(suffix=".pdf",delete=False)
#         for chunk in res.iter_content(chunk_size=8192):
#             tmp.write(chunk)
#         tmp.close()
#         return tmp.name
#     except Exception as e:
#         print(f"    ✗ Download failed: {e}")
#         return None
_download_session = None

def get_download_session() -> cf_requests.Session:
    """
    Initiating the session with all the cookies needed for downloading pdf
    """
    global _download_session
    if _download_session is not None:
        return _download_session
    session=cf_requests.Session(impersonate="chrome120")
    cookie_file=os.path.join(os.path.dirname(__file__),"justice_cookies.json")
    with open(cookie_file) as f:
        cookies=json.load(f) #loading cookies from file
    for c in cookies: #setting all the cookies to the session
        session.cookies.set(
            c["name"],
            c["value"],
            domain=c.get("domain","justice.gov"),
            path=c.get("path","/")
        )
    print(f"✓ Loaded {len(cookies)} cookies from file")
    _download_session=session
    return session


def download_pdf(url: str) -> str | None:
    """Download a PDF into a temp file, return path."""
    try:
        session = get_download_session()
        res = session.get(url, timeout=TIMEOUT, stream=True)

        # Detect HTML instead of PDF
        content_type = res.headers.get("Content-Type", "")
        first_bytes  = b""
        chunks       = []

        for chunk in res.iter_content(chunk_size=8192):
            if not first_bytes:
                first_bytes = chunk
            chunks.append(chunk)

        if b"age-verify" in first_bytes or b"<html" in first_bytes[:100]:
            print(f"    ✗ Still getting HTML — age gate not bypassed")
            return None

        if not first_bytes.startswith(b"%PDF"):
            print(f"    ✗ Not a valid PDF (starts with: {first_bytes[:20]})")
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        for chunk in chunks:
            tmp.write(chunk)
        tmp.close()
        return tmp.name

    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return None
# print(download_pdf("https://www.justice.gov/epstein/files/DataSet%204/EFTA00008320.pdf"))
#download test passed
def process_pdf(pdf_info:dict,vector_store)->bool:
    """
    Full pipeline for one PDF:
    download → extract → chunk → embed → push to Pinecone → delete

    Returns True if successful, False if failed.
    """
    filename=pdf_info["filename"]
    url=pdf_info["url"]
    dataset=pdf_info["dataset"]
    print(f"\n  📄 {filename}")
    tmp_path=download_pdf(url)
    if not tmp_path:
        return False
    try:
        raw_text=extract_text_from_pdf(tmp_path)
        if not raw_text.strip():
            return False
        print(f"    ✓ Extracted {len(raw_text):,} chars")
        #chunk text
        chunk_data=chunk_text(raw_text,filename,url,dataset)
        texts=[c["text"] for c in chunk_data]
        metadatas=[c["metadatas"] for c in chunk_data]
        vector_store.add(
            texts=texts,
            metadatas=metadatas,
            namespace=NAMESPACE
        )
        save_pdf_record(
            filename=filename,
            url=url,
            dataset=dataset,
            chunk_count=len(texts),
            char_count=len(raw_text)
        )
        return True

    except Exception as e:
        print(f"    ✗ Processing error: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
def run_scraper(dataset_pages:list[str] | None):
    """
    Main entry point.
    Scrapes all dataset pages, processes each PDF one at a time.
    Skips already-processed files using Postgres record.
    """
    pages=dataset_pages or DATASET_PAGES
    print("Loading vector store...")
    vector_store=setup_pinecone_index()
    print("✓ Vector store ready\n")
    total_processed = 0
    total_skipped   = 0
    total_failed    = 0
    for page_url in pages:
        pdf_links = scrape_pdf_links(page_url)
        if not pdf_links:
            continue
        for pdf_info in tqdm(pdf_links,desc=f"Processing {page_url}!.."):
            filename=pdf_info["filename"]
            url=pdf_info["url"]
            #Skip if aldready processed (stored in postgres)
            if is_aldready_processed(filename) or is_aldready_processed(page_url):
                total_skipped+=1
                continue
            #process the current pdf
            success=process_pdf(pdf_info,vector_store)
            if success:
                total_processed += 1
            else:
                total_failed+=1
            time.sleep(DELAY)
        print(f"\n{'─'*50}")
        print(f"✅ Scraping complete!")
        print(f"   Processed: {total_processed}")
        print(f"   Skipped:   {total_skipped} (already done)")
        print(f"   Failed:    {total_failed}")
    

if __name__ == "__main__":
    run_scraper()
        
    


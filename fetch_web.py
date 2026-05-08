import requests
from bs4 import BeautifulSoup
import os

urls = [
    "https://svelte.dev/docs/svelte/overview",
    "https://svelte.dev/docs/svelte/what-are-runes",
    "https://svelte.dev/docs/svelte/$state",
    "https://svelte.dev/docs/svelte/$derived",
    "https://svelte.dev/docs/svelte/$effect"
]

os.makedirs("docs", exist_ok=True)

for i, url in enumerate(urls):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    # 스크립트, 스타일 태그 제거
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    filename = f"docs/web_{i}.txt"

    with open(filename, "w") as f:
        f.write(text)
    print(f"{url} → {filename} ({len(text)}자)")


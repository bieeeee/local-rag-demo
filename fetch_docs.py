import wikipedia
wikipedia.set_lang("en")

pages = ["Python (programming language)", "Svelte"]
for title in pages:
    page = wikipedia.page(title)
    with open(f"docs/{title.split('(')[0].strip()}.txt", "w") as f:
        f.write(page.content)
    print(f"{title}: {len(page.content)}자 저장")

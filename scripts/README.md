# Corpus Generation Scripts

These scripts were used to generate the sample knowledge base under `docs/`.

The workflow is:

1. Download publicly available webpages from supported sources (CDC, NPS).
2. Convert HTML into Markdown.
3. Remove navigation, branding, and non-text content.
4. Perform manual review.
5. Store the adapted documents under `docs/`.

The generated Markdown files are intended for demonstration purposes only.
Always review generated documents before publishing them.

## Scripts

- `cdc_fetch.py` — scrapes curated `cdc.gov` pages.
- `nps_fetch.py` — scrapes curated `nps.gov` pages.
- `sources.yaml` — shared source list for both scrapers.

## Install

```bash
python -m pip install -r scripts/requirements-fetch.txt
```

## Configure

1. Open `scripts/cdc_fetch.py` and `scripts/nps_fetch.py`.
2. Replace `REPLACE_WITH_YOUR_REPO` and `REPLACE_WITH_YOUR_EMAIL` in
   `DEFAULT_USER_AGENT` (each scraper aborts if these placeholders remain).
3. Edit `scripts/sources.yaml` to select the pages you want. Each source
   (`cdc`, `nps`) has its own `pages` list.

## Run

```bash
# CDC pages
python scripts/cdc_fetch.py --config scripts/sources.yaml

# NPS pages
python scripts/nps_fetch.py --config scripts/sources.yaml
```

`--config scripts/sources.yaml` is the default, so it can be omitted.

Useful options (available on both scrapers):

```bash
# Preview extraction without writing files
python scripts/cdc_fetch.py --dry-run

# Regenerate existing Markdown files
python scripts/cdc_fetch.py --overwrite

# Do not retain downloaded HTML
python scripts/cdc_fetch.py --no-raw
```

## Generated output

```text
docs/**/*.md
docs/source-manifest.yaml        # nps_fetch.py
docs/cdc-source-manifest.yaml    # cdc_fetch.py
raw/cdc/**/*.html
raw/nps/**/*.html
```

Review every generated Markdown file before publishing it. The scripts
deliberately set `reuse_status: manual-review-required`; they do not make a
legal determination that an individual page is entirely public domain.

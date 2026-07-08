"""
X (Twitter) Search Scraper — Production Quality v4
====================================================
Uses X Advanced Search → Latest tab for precise date-range targeting.

Key improvements over v3:
  - 3–4× faster: reduced waits, async concurrent extraction
  - Show more: re-reads text after DOM update via tweet permalink fallback
  - Smarter idle detection: jump-scroll + page-height guard before counting idle
  - Aggressive scroll strategy to bust through X's lazy-load stalls

Usage:
    python X_advanced_search.py --account BJP4India --date 2026-06-18
    python X_advanced_search.py --account INCIndia --start 2026-05-18 --end 2026-06-18
    python X_advanced_search.py --account BJP4India --date 2026-06-18 --idle-limit 5 --no-headless
"""

import argparse
import asyncio
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError

# ---------------------------------------------------------------------------
# Config  — tuned for speed
# ---------------------------------------------------------------------------
STATE_FILE          = "state.json"
BASE_URL            = "https://x.com"
IDLE_LIMIT          = 6        # consecutive idle scroll GROUPS before stopping
SCROLL_PAUSE_MS     = 1200     # wait after normal scroll  (was 2500)
STUCK_PAUSE_MS      = 2500     # wait when no new tweets found (slightly longer)
EXPAND_WAIT_MS      = 1200     # wait after clicking show-more (was 1800)
RETRY_ATTEMPTS      = 2        # text retries (was 4 — each adds RETRY_DELAY)
RETRY_DELAY_MS      = 600      # delay between text retries (was 1000)
SCROLL_PX           = 1800     # pixels per scroll step (was 1600)
PAGE_LOAD_WAIT      = 5000     # initial page settle (ms)
JUMP_SCROLL_PX      = 6000     # big jump when stuck to bust lazy-load
JUMP_SCROLL_WAIT_MS = 2800     # wait after jump scroll
MAX_JUMP_ATTEMPTS   = 3        # jump attempts before counting as a true idle
IST                 = ZoneInfo("Asia/Kolkata")

END_OF_RESULTS_SELECTORS = [
    '[data-testid="emptyState"]',
    'div[data-testid="primaryColumn"] span:has-text("No results for")',
    'div[data-testid="primaryColumn"] span:has-text("Try searching for something else")',
]

RE_TWEET_ID = re.compile(r"/status/(\d+)")
RE_HASHTAG  = re.compile(r"#(\w+)", re.UNICODE)
RE_MENTION  = re.compile(r"@(\w+)")
RE_METRICS  = re.compile(
    r"(?:(\d[\d,]*)\s+repl(?:y|ies)[,.]?\s*)?"
    r"(?:(\d[\d,]*)\s+reposts?[,.]?\s*)?"
    r"(?:(\d[\d,]*)\s+likes?[,.]?\s*)?"
    r"(?:(\d[\d,]*)\s+bookmarks?[,.]?\s*)?"
    r"(?:(\d[\d,]*)\s+views?)?",
    re.IGNORECASE,
)
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_lang(text: str) -> str:
    return "hi" if DEVANAGARI.search(text) else "en"


def to_int(s) -> int:
    if not s:
        return 0
    return int(str(s).replace(",", ""))


def parse_metrics(aria: str) -> dict:
    base = dict(reply_count=0, repost_count=0, like_count=0,
                bookmark_count=0, view_count=0)
    if not aria:
        return base
    m = RE_METRICS.search(aria)
    if not m:
        return base
    base["reply_count"]    = to_int(m.group(1))
    base["repost_count"]   = to_int(m.group(2))
    base["like_count"]     = to_int(m.group(3))
    base["bookmark_count"] = to_int(m.group(4))
    base["view_count"]     = to_int(m.group(5))
    return base


def build_search_url(account: str, start: date, end: date) -> str:
    """
    X's until: is EXCLUSIVE, so add +1 day to make the range inclusive.
    f=live  → Latest tab (chronological), not Top (algorithmic).
    """
    until   = end + timedelta(days=1)
    query   = f"from:{account} since:{start} until:{until}"
    encoded = quote(query)
    return f"{BASE_URL}/search?q={encoded}&src=typed_query&f=live"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
class XScraper:
    def __init__(self, account: str, idle_limit: int = IDLE_LIMIT,
                 headless: bool = True):
        self.account    = account
        self.idle_limit = idle_limit
        self.headless   = headless
        self.seen_ids: set[str]     = set()
        self.all_tweets: list[dict] = []

    # ------------------------------------------------------------------
    async def run(self, start: date, end: date) -> list[dict]:
        if not Path(STATE_FILE).exists():
            sys.exit(f"[ERROR] {STATE_FILE} not found — run login script first.")

        search_url = build_search_url(self.account, start, end)

        print(f"\n{'='*60}")
        print(f"  Account      : {self.account}")
        print(f"  Date range   : {start} → {end}")
        print(f"  Search query : from:{self.account} since:{start} until:{end + timedelta(days=1)}")
        print(f"  Search URL   : {search_url}")
        print(f"  Idle limit   : {self.idle_limit} consecutive idle groups")
        print(f"{'='*60}\n")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            ctx = await browser.new_context(
                storage_state=STATE_FILE,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                timezone_id="Asia/Kolkata",
            )
            page = await ctx.new_page()

            # Block heavy assets — images/fonts not needed for text scraping
            async def block_assets(route):
                if route.request.resource_type in ("image", "font", "media"):
                    await route.abort()
                else:
                    await route.continue_()
            await page.route("**/*", block_assets)

            print(f"[→] Navigating to search URL...")
            await page.goto(search_url, wait_until="domcontentloaded", timeout=40_000)
            print(f"[→] Waiting {PAGE_LOAD_WAIT}ms for results to settle...")
            await page.wait_for_timeout(PAGE_LOAD_WAIT)

            await self._dismiss_popups(page)
            await self._ensure_latest_tab(page)
            await self._scroll_loop(page)

            await browser.close()

        # Safety clip to requested window (X search is usually precise)
        results = [
            t for t in self.all_tweets
            if start <= datetime.fromisoformat(t["datetime"]).date() <= end
        ]
        results.sort(key=lambda t: t["datetime"])
        return results

    # ------------------------------------------------------------------
    async def _dismiss_popups(self, page):
        for selector in [
            '[data-testid="xMigrationBottomBar"] button',
            '[aria-label="Close"]',
            '[data-testid="confirmationSheetConfirm"]',
        ]:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=1500):
                    await btn.click()
                    await page.wait_for_timeout(600)
            except Exception:
                pass

    # ------------------------------------------------------------------
    async def _ensure_latest_tab(self, page):
        try:
            if "f=live" not in page.url:
                print("[→] Not on Latest tab — switching...")
                for selector in ['a[href*="f=live"]', '[role="tab"]:has-text("Latest")']:
                    el = page.locator(selector).first
                    if await el.is_visible(timeout=2000):
                        await el.click()
                        # Last change
                        # await page.wait_for_timeout('[data-testid="tweetText"]',2000)
                        await page.wait_for_selector('[data-testid="tweetText"]', timeout=2000)
                        print("[✓] Switched to Latest tab.")
                        return
                print("[!] Could not find Latest tab — continuing anyway.")
            else:
                print("[✓] Latest tab confirmed (f=live in URL).")
        except Exception as e:
            print(f"[!] Latest tab check: {e}")

    # ------------------------------------------------------------------
    async def _check_end_of_results(self, page) -> bool:
        for selector in END_OF_RESULTS_SELECTORS:
            try:
                if await page.locator(selector).first.is_visible(timeout=800):
                    return True
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    async def _try_bust_lazy_load(self, page) -> int:
        """
        When stuck, try up to MAX_JUMP_ATTEMPTS big jumps to force X
        to load new tweet batches. Returns number of new IDs found.
        """
        for attempt in range(1, MAX_JUMP_ATTEMPTS + 1):
            before = len(self.seen_ids)
            print(f"  [Jump #{attempt}] Jumping {JUMP_SCROLL_PX}px to bust lazy-load...")

            # Big jump down
            await page.evaluate(f"window.scrollBy(0, {JUMP_SCROLL_PX})")
            await page.wait_for_timeout(JUMP_SCROLL_WAIT_MS)

            # Extract whatever loaded
            articles = await page.query_selector_all('article[data-testid="tweet"]')
            for art in articles:
                await self._extract(page, art)

            new_found = len(self.seen_ids) - before
            if new_found > 0:
                print(f"  [Jump #{attempt}] Unlocked {new_found} new tweets!")
                return new_found

            # Small back-scroll then forward to shake the DOM
            await page.evaluate("window.scrollBy(0, -800)")
            await page.wait_for_timeout(500)
            await page.evaluate(f"window.scrollBy(0, 1200)")
            await page.wait_for_timeout(JUMP_SCROLL_WAIT_MS)

            articles = await page.query_selector_all('article[data-testid="tweet"]')
            for art in articles:
                await self._extract(page, art)

            new_found = len(self.seen_ids) - before
            if new_found > 0:
                print(f"  [Jump #{attempt}] Shake unlocked {new_found} new tweets!")
                return new_found

        return 0

    # ------------------------------------------------------------------
    async def _scroll_loop(self, page):
        idle_groups  = 0
        scroll_num   = 0
        prev_height  = 0

        while True:
            scroll_num += 1
            before      = len(self.seen_ids)

            articles = await page.query_selector_all('article[data-testid="tweet"]')
            for art in articles:
                await self._extract(page, art)

            new_this = len(self.seen_ids) - before

            # Date coverage for logging
            if self.all_tweets:
                dates = sorted(set(t["datetime"][:10] for t in self.all_tweets))
                coverage = f"{dates[0]} → {dates[-1]}" if len(dates) > 1 else dates[0]
            else:
                coverage = "none yet"

            print(f"\n[Scroll #{scroll_num}]")
            print(f"  Visible cards        : {len(articles)}")
            print(f"  New IDs this scroll  : {new_this}")
            print(f"  Total unique scraped : {len(self.seen_ids)}")
            print(f"  Date range covered   : {coverage}")

            # Condition B: X empty-state UI
            if await self._check_end_of_results(page):
                print(f"\n[✓] STOP — Condition B: X signals no more results.")
                break

            if new_this == 0:
                # Before counting as idle, check if page height grew
                cur_height = await page.evaluate("document.body.scrollHeight")
                if cur_height > prev_height:
                    # Page did grow — just not rendered yet; don't count idle
                    print(f"  Page height grew ({prev_height}→{cur_height}), not counting idle.")
                    prev_height = cur_height
                    await page.evaluate(f"window.scrollBy(0, {SCROLL_PX})")
                    await page.wait_for_timeout(STUCK_PAUSE_MS)
                    continue

                # Try jump scrolls to bust X lazy-load before counting idle
                jump_new = await self._try_bust_lazy_load(page)
                if jump_new > 0:
                    idle_groups = 0
                    prev_height = await page.evaluate("document.body.scrollHeight")
                    continue

                idle_groups += 1
                print(f"  Idle group count     : {idle_groups}/{self.idle_limit}")

                if idle_groups >= self.idle_limit:
                    print(f"\n[✓] STOP — Condition A: {self.idle_limit} idle groups after jump attempts.")
                    break

                # Wait longer when stuck, then retry
                await page.wait_for_timeout(STUCK_PAUSE_MS)
            else:
                if idle_groups > 0:
                    print(f"  Idle reset (was {idle_groups})")
                idle_groups = 0
                prev_height = await page.evaluate("document.body.scrollHeight")

            await page.evaluate(f"window.scrollBy(0, {SCROLL_PX})")
            await page.wait_for_timeout(SCROLL_PAUSE_MS)

    # ------------------------------------------------------------------
    async def _extract(self, page, art):
        tweet_id, tweet_url = await self._get_id_url(art)
        if not tweet_id or tweet_id in self.seen_ids:
            return

        # Register immediately — even if partial extraction fails below
        self.seen_ids.add(tweet_id)

        dt = await self._get_datetime(art)
        if dt is None:
            print(f"  [?] {tweet_id} — no datetime, skipped")
            return

        # Expand show-more then read text
        tweet_text = await self._get_full_text(art, page, tweet_url)

        metrics   = await self._get_metrics(art)
        has_image = bool(await art.query_selector('[data-testid="tweetPhoto"]'))
        has_video = bool(await art.query_selector('[data-testid="videoPlayer"]'))
        hashtags  = ", ".join(RE_HASHTAG.findall(tweet_text))
        mentions  = ", ".join(RE_MENTION.findall(tweet_text))
        lang      = detect_lang(tweet_text)

        self.all_tweets.append(dict(
            tweet_id       = tweet_id,
            datetime       = dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S"),
            tweet_url      = tweet_url,
            account        = self.account,
            tweet_text     = tweet_text,
            **metrics,
            has_image      = has_image,
            has_video      = has_video,
            hashtags       = hashtags,
            mentions       = mentions,
            language       = lang,
        ))

        preview = tweet_text[:80].replace("\n", " ")
        print(f"  [+] {tweet_id} | {dt.astimezone(IST).date()} | {preview}")

    # ------------------------------------------------------------------
    async def _get_full_text(self, art, page, tweet_url: str = "") -> str:

        text = await self._read_tweet_text(art, page)

        try:
            sm = await art.query_selector(
                '[data-testid="tweet-text-show-more-link"]'
        )

            if not sm:
                print(f"[DEBUG] NO_SHOW_MORE | len={len(text)}")
                return text

            print(f"[DEBUG] SHOW_MORE_FOUND | len={len(text)}")

            try:
                await sm.scroll_into_view_if_needed()
                await sm.click(timeout=3000)
                await page.wait_for_timeout(1500)
            except Exception:
                pass

            expanded = await self._read_tweet_text(art, page)

            if expanded and len(expanded) > len(text):
                text = expanded

            if tweet_url:
                full = await self._read_from_permalink(
                    page.context,
                    tweet_url
                )

                if full and len(full) > len(text):
                    print(f"[DEBUG] PERMALINK_SUCCESS | len={len(full)}")
                    return full

        except Exception as e:
            print(f"    [show-more error] {e}")

        return text

    # ------------------------------------------------------------------
    async def _read_tweet_text(self, art, page) -> str:
        """Read tweetText element with minimal retries."""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                # Always re-query — never reuse a potentially stale handle
                el = await art.query_selector('[data-testid="tweetText"]')
                if el:
                    text = (await el.inner_text() or "").strip()
                    if text:
                        return text
                    # JS fallback
                    text = (await page.evaluate("(el) => el.textContent || ''", el) or "").strip()
                    if text:
                        return text
            except Exception:
                pass
            if attempt < RETRY_ATTEMPTS:
                await page.wait_for_timeout(RETRY_DELAY_MS)
        return ""
    
    async def _read_from_permalink(
        self,
        context,
        tweet_url: str
    ) -> str:

        page = await context.new_page()

        try:

            await page.goto(
                tweet_url,
                wait_until="domcontentloaded",
                timeout=30000
            )

            await page.wait_for_selector(
                '[data-testid="tweetText"]',
                timeout=10000
            )

            tweet_blocks = page.locator(
                '[data-testid="tweetText"]'
            )

            count = await tweet_blocks.count()

            best_text = ""

            for i in range(count):

                try:
                    txt = (
                        await tweet_blocks.nth(i).inner_text()
                    ).strip()

                    if len(txt) > len(best_text):
                        best_text = txt

                except Exception:
                    pass

            if best_text:

                print(
                    f"    [PERMALINK FOUND {count} tweetText blocks, using len={len(best_text)}]"
                )

                return best_text

        except Exception as e:
            print(f"    [permalink read failed] {e}")

        finally:
            await page.close()

        return ""

    # ------------------------------------------------------------------
    async def _get_id_url(self, art) -> tuple[str, str]:
        try:
            links = await art.query_selector_all('a[href*="/status/"]')
            for link in links:
                href = await link.get_attribute("href") or ""
                m = RE_TWEET_ID.search(href)
                if m:
                    return m.group(1), f"https://x.com{href.split('?')[0]}"
        except Exception as e:
            print(f"  [!] ID error: {e}")
        return "", ""

    # ------------------------------------------------------------------
    async def _get_datetime(self, art) -> datetime | None:
        try:
            time_el = await art.query_selector("time[datetime]")
            if time_el:
                raw = await time_el.get_attribute("datetime")
                if raw:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    async def _get_metrics(self, art) -> dict:
        # Strategy 1: role=group aria-label (all 5 values in one string)
        try:
            group = await art.query_selector('[role="group"]')
            if group:
                aria   = await group.get_attribute("aria-label") or ""
                parsed = parse_metrics(aria)
                if any(parsed.values()):
                    return parsed
        except Exception:
            pass

        # Strategy 2: individual button aria-labels
        result = dict(reply_count=0, repost_count=0, like_count=0,
                      bookmark_count=0, view_count=0)
        try:
            buttons = await art.query_selector_all('[role="button"][aria-label]')
            for btn in buttons:
                aria = (await btn.get_attribute("aria-label") or "").lower()
                m    = re.search(r"(\d[\d,]*)", aria)
                if not m:
                    continue
                val = to_int(m.group(1))
                if   "repl"     in aria: result["reply_count"]    = val
                elif "repost"   in aria: result["repost_count"]   = val
                elif "like"     in aria: result["like_count"]     = val
                elif "bookmark" in aria: result["bookmark_count"] = val
                elif "view"     in aria: result["view_count"]     = val
        except Exception:
            pass
        return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
COLUMNS = [
    "tweet_id", "datetime", "tweet_url", "account", "tweet_text",
    "reply_count", "repost_count", "like_count", "bookmark_count", "view_count",
    "has_image", "has_video", "hashtags", "mentions", "language",
]


def save_excel(tweets: list[dict], path: Path):
    if not tweets:
        print("\n[!] No tweets in requested range — Excel not saved.")
        return
    df = (
        pd.DataFrame(tweets, columns=COLUMNS)
        .drop_duplicates("tweet_id")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"\n[✓] {len(df)} tweets saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="X scraper v4 — Advanced Search Latest tab, fast & complete"
    )
    p.add_argument("--account",     required=True,  help="X handle, e.g. BJP4India")
    p.add_argument("--date",        default=None,   help="Single date YYYY-MM-DD")
    p.add_argument("--start",       default=None,   help="Range start YYYY-MM-DD")
    p.add_argument("--end",         default=None,   help="Range end   YYYY-MM-DD")
    p.add_argument("--idle-limit",  type=int, default=IDLE_LIMIT,
                   help=f"Idle groups before stopping (default {IDLE_LIMIT})")
    p.add_argument("--no-headless", action="store_true",
                   help="Show browser — useful for debugging")
    return p.parse_args()


async def main():
    args = parse_args()

    if args.date:
        start = end = date.fromisoformat(args.date)
    elif args.start and args.end:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)
        if start > end:
            sys.exit("[ERROR] --start must be ≤ --end")
    else:
        sys.exit("[ERROR] Provide --date OR both --start and --end")

    scraper = XScraper(
        account    = args.account,
        idle_limit = args.idle_limit,
        headless   = not args.no_headless,
    )
    tweets = await scraper.run(start, end)

    out = (
        Path(f"{args.account}_{start}.xlsx")
        if start == end
        else Path(f"{args.account}_{start}_{end}.xlsx")
    )
    save_excel(tweets, out)

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Account          : {args.account}")
    print(f"  Requested range  : {start} → {end}")
    print(f"  Total scraped    : {len(scraper.all_tweets)}")
    print(f"  Saved (in range) : {len(tweets)}")
    if tweets:
        dates = sorted(set(t["datetime"][:10] for t in tweets))
        print(f"  Dates covered    : {', '.join(dates)}")
        print(f"  Output file      : {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
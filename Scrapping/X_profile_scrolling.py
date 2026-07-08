"""
X (Twitter) Profile Scraper — Production Quality v2
====================================================
Uses saved Playwright state.json (logged-in session).
Collects ALL tweets visible on scroll; filters by date only at save time.

Usage:
    python X_profile_scrolling.py --account BJP4India --date 2026-06-18
    python X_profile_scrolling.py --account INCIndia --start 2026-05-18 --end 2026-06-18
    python X_profile_scrolling.py --account BJP4India --date 2026-06-18 --idle-limit 10 --no-headless
"""

import argparse
import asyncio
import re
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE_FILE      = "state.json"
BASE_URL        = "https://x.com"
IDLE_LIMIT      = 8          # consecutive idle scrolls before giving up
SCROLL_PAUSE_MS = 2500       # wait after each scroll
EXPAND_WAIT_MS  = 1800       # wait after clicking show-more
RETRY_ATTEMPTS  = 4          # retries for missing tweet text
RETRY_DELAY_MS  = 1000       # delay between text retries
SCROLL_PX       = 1600       # pixels per scroll step
PAGE_LOAD_WAIT  = 5000       # initial page settle time (ms)
IST             = ZoneInfo("Asia/Kolkata")

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


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
class XScraper:
    def __init__(
        self,
        account: str,
        start_date: date,
        end_date: date,
        idle_limit: int = IDLE_LIMIT,
        headless: bool = True,
    ):
        self.account = account
        self.start_date = start_date
        self.end_date = end_date
        self.idle_limit = idle_limit
        self.headless = headless

        self.stop_scrolling = False

        self.seen_ids = set()
        self.all_tweets = []

    async def run(self, start: date, end: date) -> list[dict]:
        if not Path(STATE_FILE).exists():
            sys.exit(f"[ERROR] {STATE_FILE} not found — run login script first.")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless)
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

            # Suppress unnecessary resource loading to speed up
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf}",
                lambda r: r.abort()
            )

            url = f"{BASE_URL}/{self.account}"
            print(f"\n[→] Opening {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=40_000)
            print(f"[→] Waiting {PAGE_LOAD_WAIT}ms for feed to settle...")
            await page.wait_for_timeout(PAGE_LOAD_WAIT)

            # Dismiss any login/cookie pop-up if present
            await self._dismiss_popups(page)

            await self._scroll_loop(page)
            await browser.close()

        # Filter to requested date range
        results = [
            t for t in self.all_tweets
            if start <= datetime.fromisoformat(t["datetime"]).date() <= end
        ]
        return results

    # ------------------------------------------------------------------
    async def _dismiss_popups(self, page):
        """Try to close common X modal overlays."""
        for selector in [
            '[data-testid="xMigrationBottomBar"] button',
            '[aria-label="Close"]',
            '[data-testid="confirmationSheetConfirm"]',
        ]:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=2000):
                    await btn.click()
                    await page.wait_for_timeout(800)
            except Exception:
                pass

    # ------------------------------------------------------------------
    async def _scroll_loop(self, page):
        idle_scrolls = 0
        scroll_num   = 0

        while True:
            scroll_num  += 1
            before       = len(self.seen_ids)

            articles = await page.query_selector_all('article[data-testid="tweet"]')
            print(f"\n[Scroll #{scroll_num}] Visible cards: {len(articles)}")

            for art in articles:
                await self._extract(page, art)

                if self.stop_scrolling:
                    break

            if self.stop_scrolling:
                print("\n[✓] Target date range reached.")
                break

            new_this = len(self.seen_ids) - before
            print(f"  New IDs this scroll  : {new_this}")
            print(f"  Total unique scraped : {len(self.seen_ids)}")

            if new_this == 0:
                idle_scrolls += 1
                print(f"  Idle count           : {idle_scrolls}/{self.idle_limit}")
            else:
                idle_scrolls = 0

            if idle_scrolls >= self.idle_limit:
                print(f"\n[✓] Done — {self.idle_limit} consecutive idle scrolls.")
                break

            await page.evaluate(f"window.scrollBy(0, {SCROLL_PX})")
            await page.wait_for_timeout(SCROLL_PAUSE_MS)

    # ------------------------------------------------------------------
    async def _extract(self, page, art):
        # ── 1. Tweet ID & URL ──────────────────────────────────────────
        tweet_id, tweet_url = await self._get_id_url(art)
        if not tweet_id:
            return
        if tweet_id in self.seen_ids:
            return

        # ── 2. Datetime ────────────────────────────────────────────────
        dt = await self._get_datetime(art)

        tweet_date = dt.astimezone(IST).date()

        # We have gone older than the requested range
        if tweet_date < self.start_date:
            print(
                f"Reached {tweet_date}, older than requested "
                f"{self.start_date}. Stopping..."
            )
            self.stop_scrolling = True
            return

        if dt is None:
            # Still register the ID so we don't keep retrying on scroll
            self.seen_ids.add(tweet_id)
            print(f"  [?] {tweet_id} — could not parse datetime, skipped")
            return

        # ── 3. Expand show-more ────────────────────────────────────────
        await self._expand_show_more(art, page)

        # ── 4. Tweet text ──────────────────────────────────────────────
        tweet_text = await self._get_text(art, page)

        # ── 5. Metrics ────────────────────────────────────────────────
        metrics = await self._get_metrics(art)

        # ── 6. Media ──────────────────────────────────────────────────
        has_image = bool(await art.query_selector('[data-testid="tweetPhoto"]'))
        has_video = bool(await art.query_selector('[data-testid="videoPlayer"]'))

        # ── 7. Derived fields ─────────────────────────────────────────
        hashtags = ", ".join(RE_HASHTAG.findall(tweet_text))
        mentions = ", ".join(RE_MENTION.findall(tweet_text))
        lang     = detect_lang(tweet_text)

        self.seen_ids.add(tweet_id)
        if self.start_date <= tweet_date <= self.end_date:

            self.all_tweets.append(dict(
                tweet_id=tweet_id,
                datetime=dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S"),
                tweet_url=tweet_url,
                account=self.account,
                tweet_text=tweet_text,
                **metrics,
                has_image=has_image,
                has_video=has_video,
                hashtags=hashtags,
                mentions=mentions,
                language=lang,
            ))

        preview = tweet_text[:80].replace("\n", " ")
        print(f"  [+] {tweet_id} | {dt.astimezone(IST).date()} | {preview}")

    # ------------------------------------------------------------------
    async def _get_id_url(self, art) -> tuple[str, str]:
        """
        Find tweet ID from any <a href="/user/status/ID"> inside the article.
        Try multiple selector strategies.
        """
        try:
            # Strategy 1: direct link with /status/ in href
            links = await art.query_selector_all('a[href*="/status/"]')
            for link in links:
                href = await link.get_attribute("href") or ""
                m = RE_TWEET_ID.search(href)
                if m:
                    tweet_id = m.group(1)
                    # Build canonical URL
                    # href is like /BJP4India/status/123456
                    tweet_url = f"https://x.com{href.split('?')[0]}"
                    return tweet_id, tweet_url
        except Exception as e:
            print(f"  [!] ID extraction error: {e}")
        return "", ""

    # ------------------------------------------------------------------
    async def _get_datetime(self, art) -> datetime | None:
        try:
            time_el = await art.query_selector("time[datetime]")
            if time_el:
                raw = await time_el.get_attribute("datetime")
                if raw:
                    # e.g. "2026-06-18T10:30:00.000Z"
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    async def _expand_show_more(self, art, page):
        try:
            sm = await art.query_selector('[data-testid="tweet-text-show-more-link"]')
            if sm:
                await sm.scroll_into_view_if_needed()
                await sm.click()
                await page.wait_for_timeout(EXPAND_WAIT_MS)
        except Exception:
            pass

    # ------------------------------------------------------------------
    async def _get_text(self, art, page) -> str:
        """
        Read tweetText span with retries.
        Falls back to collecting all inner spans if selector gives empty.
        """
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                el = await art.query_selector('[data-testid="tweetText"]')
                if el:
                    text = (await el.inner_text() or "").strip()
                    if text:
                        return text
                    # Try evaluating text content via JS
                    text = await page.evaluate(
                        "(el) => el.textContent || ''", el
                    )
                    if text and text.strip():
                        return text.strip()
            except Exception:
                pass
            if attempt < RETRY_ATTEMPTS:
                await page.wait_for_timeout(RETRY_DELAY_MS)
        return ""

    # ------------------------------------------------------------------
    async def _get_metrics(self, art) -> dict:
        # Strategy 1: role=group aria-label
        try:
            group = await art.query_selector('[role="group"]')
            if group:
                aria = await group.get_attribute("aria-label") or ""
                parsed = parse_metrics(aria)
                if any(parsed.values()):
                    return parsed
        except Exception:
            pass

        # Strategy 2: individual aria-labels on buttons
        result = dict(reply_count=0, repost_count=0, like_count=0,
                      bookmark_count=0, view_count=0)
        try:
            buttons = await art.query_selector_all('[role="button"][aria-label]')
            for btn in buttons:
                aria = (await btn.get_attribute("aria-label") or "").lower()
                m = re.search(r"(\d[\d,]*)", aria)
                if not m:
                    continue
                val = to_int(m.group(1))
                if "repl" in aria:
                    result["reply_count"] = val
                elif "repost" in aria:
                    result["repost_count"] = val
                elif "like" in aria:
                    result["like_count"] = val
                elif "bookmark" in aria:
                    result["bookmark_count"] = val
                elif "view" in aria:
                    result["view_count"] = val
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
        print("\n[!] No tweets in date range — Excel not saved.")
        return
    df = pd.DataFrame(tweets, columns=COLUMNS).drop_duplicates("tweet_id")
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"\n[✓] {len(df)} tweets saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--account",    required=True)
    p.add_argument("--date",       default=None,  help="YYYY-MM-DD")
    p.add_argument("--start",      default=None,  help="YYYY-MM-DD")
    p.add_argument("--end",        default=None,  help="YYYY-MM-DD")
    p.add_argument("--idle-limit", type=int, default=IDLE_LIMIT)
    p.add_argument("--no-headless", action="store_true",
                   help="Show browser window (useful for debugging)")
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
        sys.exit("[ERROR] Provide --date OR --start + --end")

    print(f"Account    : {args.account}")
    print(f"Date range : {start} → {end}")
    print(f"Idle limit : {args.idle_limit}")
    print(f"Headless   : {not args.no_headless}")

    scraper = XScraper(
        account=args.account,
        start_date=start,
        end_date=end,
        idle_limit=args.idle_limit,
        headless=not args.no_headless,
    )
    tweets = await scraper.run(start, end)

    if start == end:
        out = Path(f"{args.account}_{start}.xlsx")
    else:
        out = Path(f"{args.account}_{start}_{end}.xlsx")

    save_excel(tweets, out)

    print(f"\n=== Final Summary ===")
    print(f"Total scraped (all dates) : {len(scraper.all_tweets)}")
    print(f"Saved in requested range  : {len(tweets)}")


if __name__ == "__main__":
    asyncio.run(main())
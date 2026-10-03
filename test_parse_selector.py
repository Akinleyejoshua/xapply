import asyncio
from config import Settings
from browser_bot import StealthBrowser
from models import JobPosting, ASHBY
from job_search import _first_text, DESCRIPTION_SELECTORS
from discovery import Database

class DummyGate: pass

async def main():
    browser = StealthBrowser(Settings(), DummyGate())
    await browser.start()
    page = browser.page
    await browser.goto(page, "https://jobs.ashbyhq.com/stickermule/2f01bd23-9eda-446a-a56a-b530d84cb9bb")
    
    # Let's try Ashby selectors
    selectors = DESCRIPTION_SELECTORS[ASHBY]
    print("ASHBY SELECTORS:", selectors)
    for sel in selectors:
        text = await _first_text(page, (sel,), min_len=200)
        print(f"SELECTOR {sel}: len {len(text)}")

    print("UNKNOWN SELECTORS:")
    for sel in ("main", "article", "body"):
        text = await _first_text(page, (sel,), min_len=200)
        print(f"SELECTOR {sel}: len {len(text)}")

asyncio.run(main())

import asyncio
from config import Settings
from browser_bot import StealthBrowser
from models import JobPosting
from job_search import extract_posting
from discovery import Database

class DummyGate: pass

async def main():
    browser = StealthBrowser(Settings(), DummyGate())
    await browser.start()
    page = browser.page
    job = JobPosting.from_url("https://jobs.ashbyhq.com/stickermule/2f01bd23-9eda-446a-a56a-b530d84cb9bb", source="urls")
    job = await extract_posting(browser, page, job)
    print("DESC LEN:", len(job.description))
    print("DESC CONTENT:", repr(job.description))

asyncio.run(main())

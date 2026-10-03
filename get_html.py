import asyncio
from config import Settings
from browser_bot import StealthBrowser

class DummyGate: pass

async def main():
    browser = StealthBrowser(Settings(), DummyGate())
    await browser.start()
    page = browser.page
    await browser.goto(page, "https://jobs.ashbyhq.com/stickermule/2f01bd23-9eda-446a-a56a-b530d84cb9bb")
    await asyncio.sleep(2)
    html = await page.evaluate("() => document.body.innerHTML")
    with open("ashby.html", "w") as f:
        f.write(html)
    await browser.stop()
asyncio.run(main())

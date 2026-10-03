import asyncio
from config import Settings
from browser_bot import StealthBrowser

class DummyGate: pass

async def main():
    browser = StealthBrowser(Settings(), DummyGate())
    await browser.start()
    page = browser.page
    await browser.goto(page, "https://jobs.ashbyhq.com/stickermule/2f01bd23-9eda-446a-a56a-b530d84cb9bb/application")
    await asyncio.sleep(2)
    # Check if we can find the overview tab and click it
    tab = page.get_by_role("tab", name="Overview")
    if await tab.count():
        print("Tab found!")
        await browser.human_click(tab.first)
        await asyncio.sleep(1)
        vis = await page.locator("div[class*='description' i]").first.is_visible()
        print("DESCRIPTION VISIBLE:", vis)
    else:
        print("Tab not found!")
    await browser.stop()
asyncio.run(main())

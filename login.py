from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)

    context = browser.new_context()

    page = context.new_page()

    page.goto("https://x.com/login")

    print("Login manually.")
    print("After login press ENTER in terminal.")

    input()

    context.storage_state(path="state.json")

    browser.close()
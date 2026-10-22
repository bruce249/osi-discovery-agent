"""End-to-end check of the deployed notebook in a real headless browser (runs on Modal, not locally).

    modal run tests/e2e_live.py
Clicks the copilot buttons and saves screenshots to the volume under results/e2e/.
"""
import modal

URL = "https://vignesh-82505--osi-discovery-agent-notebook.us-east.modal.run/"
app = modal.App("osi-e2e")
vol = modal.Volume.from_name("osi-discovery-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("playwright==1.47.0")
         .run_commands("playwright install --with-deps chromium"))


@app.function(image=image, volumes={"/vol": vol}, timeout=900)
def check(url: str = URL):
    import os

    from playwright.sync_api import sync_playwright

    out = "/vol/results/e2e"
    os.makedirs(out, exist_ok=True)
    log = []

    def shot(page, name):
        page.screenshot(path=f"{out}/{name}.png", full_page=True)

    def stats(page):
        txt = page.inner_text("body")
        i = txt.find("Current run:")
        return txt[i:i + 160].replace("\n", " ") if i >= 0 else "(no banner)"

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1300, "height": 1000})
        page.goto(url, wait_until="networkidle", timeout=180_000)
        page.get_by_text("Suggest next batch").first.wait_for(timeout=180_000)
        page.wait_for_timeout(3000)
        shot(page, "0_loaded")
        log.append(("loaded", stats(page)))
        body = page.inner_text("body")
        log.append(("new sections", {k: k in body for k in (
            "Same pattern in two domains", "Inside the simulator's mind", "Third check", "AUC 0.92",
            "Opening the box further", "nullspace", "redox state", "One network, two hidden skills", "Li2ZrBr6", "framework")},
            "pending placeholders:", [k for k in ("figure pending", "probe pending", "campaigns pending",
                                                   "phonons pending") if k in body]))
        for name, anchor in (("4_cross_domain", "Same pattern in two domains"),
                             ("5_latent", "Inside the simulator's mind"), ("6_phonons", "Third check"), ("7_fe", "Same simulator, second property")):
            el = page.get_by_text(anchor).first
            el.scroll_into_view_if_needed()
            page.wait_for_timeout(800)
            page.screenshot(path=f"{out}/{name}.png")

        page.get_by_role("button", name="Suggest next batch").click()
        page.get_by_text("what the LLM picked and why").first.wait_for(timeout=180_000)
        page.wait_for_timeout(1500)
        log.append(("after suggest", stats(page), "round 1" in page.inner_text("body")))
        shot(page, "1_after_suggest")

        page.get_by_role("button", name="New run (fresh shuffle)").click()
        page.get_by_text("Ready. Press").first.wait_for(timeout=120_000)
        page.wait_for_timeout(1500)
        log.append(("after new run", stats(page), "Your runs this session" in page.inner_text("body")))
        shot(page, "2_after_new_run")

        page.get_by_role("button", name="Suggest next batch").click()
        page.get_by_text("what the LLM picked and why").first.wait_for(timeout=180_000)
        page.wait_for_timeout(1500)
        log.append(("suggest on new run", stats(page)))

        page.get_by_role("button", name="Restart (same settings)").click()
        page.get_by_text("Ready. Press").first.wait_for(timeout=120_000)
        page.wait_for_timeout(1500)
        log.append(("after restart", stats(page)))
        shot(page, "3_after_restart")
        b.close()
    vol.commit()
    for row in log:
        print("E2E", row)

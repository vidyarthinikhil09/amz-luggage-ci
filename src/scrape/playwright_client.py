from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import Browser, Page, Playwright, sync_playwright


@dataclass(frozen=True)
class BrowserConfig:
    headless: bool
    user_agent: str


class PlaywrightBrowser:
    def __init__(self, cfg: BrowserConfig):
        self._cfg = cfg
        self._pw: Playwright | None = None
        self._browser: Browser | None = None

    def __enter__(self) -> "PlaywrightBrowser":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._cfg.headless)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def new_page(self) -> Page:
        assert self._browser is not None
        context = self._browser.new_context(user_agent=self._cfg.user_agent, viewport={"width": 1400, "height": 900})
        page = context.new_page()
        page.set_default_timeout(45_000)
        return page

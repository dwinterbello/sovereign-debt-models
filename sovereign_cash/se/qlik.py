"""Minimal client for Riksgälden's public QlikView AJAX server (statistics pages are QlikView iframes).

The statistics pages on riksgalden.se embed ``/QvAJAXZfc/opendoc.htm?document=Webb_Upplaning.qvw``.
There is no file download or API; the same XML protocol the browser speaks returns table objects.
"""

from __future__ import annotations

import html
import random
import re
import string
import time
from urllib.parse import quote

import requests

BASE = "https://www.riksgalden.se/QvAjaxZfc/QvsViewClient.aspx"
DOC = "Webb_Upplaning.qvw"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"


class Qlik:
    def __init__(self, view: str = DOC, timeout: int = 120):
        self.view = view
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.mark = ""
        self.stamp = ""

    def post(self, inner: str, *, start: bool = False) -> str:
        key = "".join(random.choices(string.ascii_letters + string.digits, k=16))
        url = f"{BASE}?mark={self.mark}&view={quote(self.view)}&platform=browser.chrome&dpi=96&xrfkey={key}"
        body = (f'<update mark="{self.mark}" stamp="{self.stamp}" cookie="true" scope="Document" '
                f'view="{self.view}" ident="null"{" start=" + chr(34) + "document" + chr(34) if start else ""}>'
                f'{inner}</update>')
        r = self.s.post(url, data=body.encode("utf-8"), timeout=self.timeout,
                        headers={"xrfkey": key, "Content-Type": "text/plain;charset=UTF-8"})
        r.raise_for_status()
        m = re.search(r'<result[^>]*mark="([^"]*)" stamp="([^"]*)"', r.text)
        if m:
            self.mark, self.stamp = m.group(1), m.group(2)
        return r.text

    def objects(self) -> list[tuple[str, str, str]]:
        t = self.post('<info name="Document.SheetObjects" />')
        return re.findall(r'<object id="([^"]+)" type="([^"]+)" caption="([^"]*)"', t)

    def poll(self, obj: str, extra: str = "", tries: int = 10) -> str:
        """Poll a sheet object until the server has computed it (first answer is mode="invalid")."""
        for _ in range(tries):
            t = self.post(f'<poll name="{obj}" {extra}/>')
            m = re.search(rf'<group name="{re.escape(obj)}"[^>]*mode="([a-z]+)"', t)
            if m and m.group(1) == "enabled":
                return t
            time.sleep(0.6)
        raise RuntimeError(f"{obj}: not enabled after {tries} polls")

    def table(self, bookmark: str, obj: str, page: int = 40) -> list[list[str]]:
        """Every cell of a pivot / straight table as columns of element texts (pages across columns)."""
        self.post(f'<set name="bookmark-apply" docaction="{bookmark}"/>', start=True)
        t = self.poll(obj)
        tot = re.search(r'totalsize="(\d+):(\d+)"', t)
        ncol, nrow = (int(tot.group(1)), int(tot.group(2))) if tot else (0, 0)
        cols: dict[str, list[str]] = {}
        x = 0
        while True:
            for m in re.finditer(r'<value name="C(\d+)"[^>/]*>(.*?)</value>', t, re.S):
                cells = [html.unescape(e) for e in re.findall(r'<element [^>]*?text="([^"]*)"', m.group(2))]
                if cells:
                    cols.setdefault(cells[0] if int(m.group(1)) else "__labels__", cells)
            x += page
            if x >= ncol:
                break
            t = self.poll(obj, f'offset="{x}:0"')
        return list(cols.values())

    def rows(self, bookmark: str, obj: str, page: int = 40) -> tuple[list[str], list[list[str]]]:
        """A straight table as (header, rows), paging down the rows."""
        self.post(f'<set name="bookmark-apply" docaction="{bookmark}"/>', start=True)
        t = self.poll(obj)
        tot = re.search(r'totalsize="(\d+):(\d+)"', t)
        nrow = int(tot.group(2)) if tot else 0
        header: list[str] = []
        out: list[list[str]] = []
        y = 0
        while True:
            cols = [[html.unescape(e) for e in re.findall(r'<element [^>]*?text="([^"]*)"', m.group(2))]
                    for m in re.finditer(r'<value name="C(\d+)"[^>/]*>(.*?)</value>', t, re.S)]
            if not cols:
                break
            if not header:
                header = [c[0] if c else "" for c in cols]
            n = max(len(c) for c in cols)
            for i in range(n):
                r = [c[i] if i < len(c) else "" for c in cols]
                if r == header:
                    continue
                out.append(r)
            y += page
            if y >= nrow:
                break
            t = self.poll(obj, f'offset="0:{y}"')
        return header, out

    def get(self, name: str) -> bytes:
        r = self.s.get(f"{BASE}?public=only&size=long&name={name}", timeout=self.timeout)
        r.raise_for_status()
        return r.content

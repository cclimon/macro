"""server.py

Macro FX Journal — backend. Python 3.9, stdlib only, no build step.

Run: python3 server.py
Listens on 127.0.0.1:3170 AND [::1]:3170 (Safari resolves localhost to IPv6;
a v4-only bind gives "Load failed"). Loopback only — never a network
interface, since filed content is confidential.
"""
import json
import os
import re
import socket
import threading
from datetime import date as date_cls
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import storage
import model
import git_ops
from tagging import tag_text

PORT = 3170
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")


# ---------------------------------------------------------------------------
# Dual-loopback server: bind 127.0.0.1 and ::1 both on PORT
# ---------------------------------------------------------------------------
class DualStackThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    request_queue_size = 32

    def server_bind(self):
        # Disable IPV6_V6ONLY so this socket ALSO accepts IPv4 on some
        # platforms; however this is not reliable cross-platform, so we
        # additionally spin up a second, real IPv4 server (see main()).
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()


class IPv4ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "MacroFXJournal/1.0"

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    # -- helpers --------------------------------------------------------
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _serve_static(self, url_path: str):
        if url_path == "/" or url_path == "":
            url_path = "/index.html"
        # path-traversal protection
        safe_rel = os.path.normpath(url_path).lstrip("/\\")
        full_path = os.path.normpath(os.path.join(PUBLIC_DIR, safe_rel))
        if not full_path.startswith(os.path.normpath(PUBLIC_DIR)):
            self.send_response(403)
            self.end_headers()
            return
        if not os.path.isfile(full_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        content_type = "application/octet-stream"
        if full_path.endswith(".html"):
            content_type = "text/html; charset=utf-8"
        elif full_path.endswith(".css"):
            content_type = "text/css; charset=utf-8"
        elif full_path.endswith(".js"):
            content_type = "application/javascript; charset=utf-8"

        with open(full_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routing ----------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            self._send_json(200, {"mockMode": model.MOCK_MODE})
            return
        if parsed.path == "/api/day":
            qs = parse_qs(parsed.query)
            date_str = (qs.get("date") or [""])[0]
            if not storage.is_valid_date(date_str):
                self._send_json(400, {"error": "invalid or missing date"})
                return
            day = storage.load_day(date_str)
            self._send_json(200, day)
            return
        self._serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/file":
            body = self._read_json_body()
            date_str = body.get("date", "")
            text = body.get("text", "")
            allow_closed = bool(body.get("allowClosed", False))

            if not storage.is_valid_date(date_str):
                self._send_json(400, {"error": "invalid or missing date"})
                return
            if not isinstance(text, str) or not text.strip():
                self._send_json(400, {"error": "text is empty"})
                return

            day = storage.load_day(date_str)
            if day["status"] == "closed" and not allow_closed:
                self._send_json(409, {"error": "day is closed", "status": "closed"})
                return

            # trim leading/trailing whitespace only; interior content untouched
            trimmed = text.strip()

            tags = tag_text(trimmed)
            filed = model.anonymise(trimmed)
            entry = storage.append_entry(date_str, trimmed, filed, tags)

            day = storage.load_day(date_str)
            self._send_json(200, {
                "entry": entry,
                "count": len(day["entries"]),
                "status": day["status"],
            })
            return

        if parsed.path == "/api/ask":
            body = self._read_json_body()
            date_str = body.get("date", "")
            query = (body.get("query") or "").strip()

            if not storage.is_valid_date(date_str):
                self._send_json(400, {"error": "invalid or missing date"})
                return
            if not query:
                self._send_json(400, {"error": "query is empty"})
                return

            day = storage.load_day(date_str)
            currency = self._resolve_query_currency(query)

            if currency:
                entries = storage.entries_for_currency(day, currency)
                if not entries:
                    answer = f"Nothing filed for {currency} on {date_str}."
                else:
                    answer = model.compose(currency, entries, date_str)
            elif query.lower() in ("summary", "all"):
                if not day["entries"]:
                    answer = f"Nothing filed on this day."
                else:
                    groups = storage.group_by_primary_tag(day)
                    parts = []
                    for ccy, entries in groups.items():
                        parts.append(f"## {ccy}\n" + model.compose(ccy, entries, date_str))
                    answer = "\n\n".join(parts)
            else:
                # free-form question over the whole day
                if not day["entries"]:
                    answer = "Nothing filed on this day."
                else:
                    answer = model.compose("GENERAL", day["entries"], date_str)

            self._send_json(200, {
                "answer": answer,
                "query": query,
                "date": date_str,
                "currency": currency,
            })
            return

        if parsed.path == "/api/eod":
            body = self._read_json_body()
            date_str = body.get("date", "")

            if not storage.is_valid_date(date_str):
                self._send_json(400, {"error": "invalid or missing date"})
                return

            day = storage.load_day(date_str)
            if day["status"] == "closed":
                self._send_json(409, {"error": "day already closed"})
                return
            if not day["entries"]:
                self._send_json(409, {"error": "no entries filed for this day"})
                return

            groups = storage.group_by_primary_tag(day)
            sections = []
            for ccy in sorted(groups.keys()):
                sections.append(f"## {ccy}\n\n{model.compose(ccy, groups[ccy], date_str)}")
            report = f"# Macro FX Journal — {date_str}\n\n" + "\n\n".join(sections) + "\n"

            report_file = storage.write_report(date_str, report)
            storage.close_day(date_str)
            day_file = storage._day_path(date_str)  # noqa: SLF001 — internal, same module family

            git_status = git_ops.commit_and_push(
                [report_file, day_file], f"EOD {date_str}"
            )

            self._send_json(200, {
                "report": report,
                "path": os.path.relpath(report_file, os.path.dirname(__file__)),
                "status": "closed",
                "git": git_status,
                "count": len(day["entries"]),
            })
            return

        self._send_json(404, {"error": "not found"})

    @staticmethod
    def _resolve_query_currency(query: str) -> str | None:
        """
        'summary GBP' or bare 'GBP' -> GBP. Returns None if the query doesn't
        resolve to a single currency request.
        """
        q = query.strip()
        m = re.match(r"^summary\s+([A-Za-z]{3})$", q, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.match(r"^([A-Za-z]{3})$", q)
        if m and m.group(1).upper() in __import__("tagging").ISO_CODES:
            return m.group(1).upper()
        return None


def main():
    v6_server = None
    v4_server = None
    try:
        v6_server = DualStackThreadingHTTPServer(("::1", PORT), Handler)
    except OSError as e:
        print(f"Could not bind ::1:{PORT} ({e}); continuing with IPv4 only.")

    try:
        v4_server = IPv4ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        print(f"Could not bind 127.0.0.1:{PORT} ({e}).")

    if not v6_server and not v4_server:
        raise SystemExit(f"Failed to bind port {PORT} on either loopback.")

    threads = []
    if v6_server:
        t = threading.Thread(target=v6_server.serve_forever, daemon=True)
        t.start()
        threads.append(t)
        print(f"Listening on http://[::1]:{PORT}")
    if v4_server:
        t = threading.Thread(target=v4_server.serve_forever, daemon=True)
        t.start()
        threads.append(t)
        print(f"Listening on http://127.0.0.1:{PORT}")

    print(f"MOCK_MODE={model.MOCK_MODE}")
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()

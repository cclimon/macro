(function () {
  "use strict";

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  const state = {
    date: todayStr(),
    day: { date: null, status: "open", entries: [] },
    mockMode: true,
    expandedSeq: null,
  };

  function todayStr() {
    const d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }

  function addDays(dateStr, delta) {
    const [y, m, d] = dateStr.split("-").map(Number);
    const dt = new Date(Date.UTC(y, m - 1, d));
    dt.setUTCDate(dt.getUTCDate() + delta);
    const ny = dt.getUTCFullYear();
    const nm = String(dt.getUTCMonth() + 1).padStart(2, "0");
    const nd = String(dt.getUTCDate()).padStart(2, "0");
    return `${ny}-${nm}-${nd}`;
  }

  function weekdayName(dateStr) {
    const [y, m, d] = dateStr.split("-").map(Number);
    const dt = new Date(Date.UTC(y, m - 1, d));
    return dt.toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" });
  }

  // ------------------------------------------------------------------
  // Elements
  // ------------------------------------------------------------------
  const el = {
    mockBadge: document.getElementById("mockBadge"),
    prevDay: document.getElementById("prevDay"),
    nextDay: document.getElementById("nextDay"),
    weekday: document.getElementById("weekday"),
    datePicker: document.getElementById("datePicker"),
    todayBtn: document.getElementById("todayBtn"),
    statusPill: document.getElementById("statusPill"),
    entryCount: document.getElementById("entryCount"),
    closedBanner: document.getElementById("closedBanner"),
    tabJournal: document.getElementById("tabJournal"),
    tabAsk: document.getElementById("tabAsk"),
    paneJournal: document.getElementById("paneJournal"),
    paneAsk: document.getElementById("paneAsk"),
    pasteBox: document.getElementById("pasteBox"),
    fileBtn: document.getElementById("fileBtn"),
    charCount: document.getElementById("charCount"),
    filedFlash: document.getElementById("filedFlash"),
    inlineError: document.getElementById("inlineError"),
    tally: document.getElementById("tally"),
    entriesBox: document.getElementById("entriesBox"),
    entriesEmpty: document.getElementById("entriesEmpty"),
    entriesList: document.getElementById("entriesList"),
    askInput: document.getElementById("askInput"),
    askHistory: document.getElementById("askHistory"),
  };

  // ------------------------------------------------------------------
  // API
  // ------------------------------------------------------------------
  async function apiGetConfig() {
    const r = await fetch("/api/config");
    return r.json();
  }
  async function apiGetDay(dateStr) {
    const r = await fetch(`/api/day?date=${encodeURIComponent(dateStr)}`);
    return r.json();
  }
  async function apiFile(dateStr, text, allowClosed) {
    const r = await fetch("/api/file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date: dateStr, text, allowClosed: !!allowClosed }),
    });
    return { status: r.status, body: await r.json() };
  }
  async function apiAsk(dateStr, query) {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date: dateStr, query }),
    });
    return { status: r.status, body: await r.json() };
  }
  async function apiEod(dateStr) {
    const r = await fetch("/api/eod", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date: dateStr }),
    });
    return { status: r.status, body: await r.json() };
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------
  function renderDayHeader() {
    el.datePicker.value = state.date;
    el.weekday.textContent = weekdayName(state.date);
    el.todayBtn.classList.toggle("hidden", state.date === todayStr());

    const closed = state.day.status === "closed";
    el.statusPill.textContent = closed ? "CLOSED" : "OPEN";
    el.statusPill.classList.toggle("open", !closed);
    el.statusPill.classList.toggle("closed", closed);
    el.entryCount.textContent = `${state.day.entries.length} ${state.day.entries.length === 1 ? "entry" : "entries"}`;
    el.closedBanner.classList.toggle("hidden", !closed);
  }

  function renderTally() {
    el.tally.textContent = "";
    const counts = {};
    for (const e of state.day.entries) {
      const tags = e.tags && e.tags.length ? e.tags : ["—"];
      const seen = new Set();
      for (const t of tags) {
        if (seen.has(t)) continue;
        seen.add(t);
        counts[t] = (counts[t] || 0) + 1;
      }
    }
    const order = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    for (const tag of order) {
      const chip = document.createElement("span");
      chip.className = "chip";
      const b = document.createElement("b");
      b.textContent = String(counts[tag]);
      chip.appendChild(document.createTextNode(tag + " "));
      chip.appendChild(b);
      el.tally.appendChild(chip);
    }
  }

  function renderEntries() {
    el.entriesList.textContent = "";
    const entries = state.day.entries;
    el.entriesEmpty.classList.toggle("hidden", entries.length > 0);

    const sorted = [...entries].sort((a, b) => b.seq - a.seq);
    for (const entry of sorted) {
      const row = document.createElement("div");
      row.className = "entry-row";
      row.dataset.seq = String(entry.seq);
      if (state.expandedSeq === entry.seq) row.classList.add("expanded");

      const chev = document.createElement("div");
      chev.className = "chev";
      chev.textContent = state.expandedSeq === entry.seq ? "\u25B2" : "\u25BC";

      const seq = document.createElement("div");
      seq.className = "seq";
      seq.textContent = `#${entry.seq}`;

      const time = document.createElement("div");
      time.className = "time";
      time.textContent = formatTime(entry.ts);

      const tagsline = document.createElement("div");
      tagsline.className = "tagsline";
      if (entry.tags && entry.tags.length) {
        for (const t of entry.tags) {
          const span = document.createElement("span");
          span.className = "tag";
          span.textContent = `{${t}}`;
          tagsline.appendChild(span);
        }
      } else {
        const span = document.createElement("span");
        span.className = "notag";
        span.textContent = "\u2014";
        tagsline.appendChild(span);
      }
      const snip = document.createElement("span");
      snip.className = "snip";
      snip.textContent = (entry.filed || "").replace(/\s+/g, " ").trim();
      tagsline.appendChild(snip);

      row.appendChild(chev);
      row.appendChild(seq);
      row.appendChild(time);
      row.appendChild(tagsline);

      row.addEventListener("click", () => {
        state.expandedSeq = state.expandedSeq === entry.seq ? null : entry.seq;
        renderEntries();
      });

      el.entriesList.appendChild(row);

      const body = document.createElement("div");
      body.className = "entry-body";
      const pre = document.createElement("pre");
      pre.textContent = entry.filed || "";
      body.appendChild(pre);
      el.entriesList.appendChild(body);
    }

    updateEntriesFade();
  }

  function formatTime(iso) {
    if (!iso) return "";
    const m = iso.match(/T(\d{2}):(\d{2}):(\d{2})/);
    return m ? `${m[1]}:${m[2]}:${m[3]}` : iso;
  }

  function updateEntriesFade() {
    const box = el.entriesBox;
    const wrap = box.parentElement;
    const canScroll = box.scrollHeight > box.clientHeight;
    const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 1;
    wrap.classList.toggle("has-more", canScroll && !atBottom);
  }

  function scrollEntriesToTop() {
    el.entriesBox.scrollTop = 0;
    updateEntriesFade();
  }

  // ------------------------------------------------------------------
  // Load / refresh day
  // ------------------------------------------------------------------
  async function loadDay(dateStr) {
    state.date = dateStr;
    state.expandedSeq = null;
    state.day = await apiGetDay(dateStr);
    renderDayHeader();
    renderTally();
    renderEntries();
  }

  // ------------------------------------------------------------------
  // Filing
  // ------------------------------------------------------------------
  let flashTimer = null;

  function showInlineError(msg) {
    el.inlineError.textContent = msg;
    el.inlineError.classList.remove("hidden");
  }
  function clearInlineError() {
    el.inlineError.classList.add("hidden");
    el.inlineError.textContent = "";
  }

  function flashBox(kind) {
    el.pasteBox.classList.remove("flash-amber", "flash-red");
    // force reflow so the class re-triggers if flashed twice quickly
    void el.pasteBox.offsetWidth;
    el.pasteBox.classList.add(kind === "red" ? "flash-red" : "flash-amber");
    setTimeout(() => el.pasteBox.classList.remove("flash-amber", "flash-red"), 350);
  }

  async function doFile(allowClosed) {
    const text = el.pasteBox.value;
    if (!text.trim()) {
      flashBox("red");
      return;
    }
    clearInlineError();

    const { status, body } = await apiFile(state.date, text, allowClosed);

    if (status === 409 && !allowClosed) {
      const ok = window.confirm("This day is closed. File anyway?");
      if (ok) {
        await doFile(true);
      }
      return;
    }

    if (status !== 200) {
      showInlineError(body.error || `Filing failed (status ${status}).`);
      return;
    }

    el.pasteBox.value = "";
    el.pasteBox.focus();
    flashBox("amber");

    el.filedFlash.textContent = `FILED #${body.entry.seq}`;
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => { el.filedFlash.textContent = ""; }, 1200);

    await loadDay(state.date);
    scrollEntriesToTop();
  }

  el.fileBtn.addEventListener("click", () => doFile(false));

  el.pasteBox.addEventListener("input", () => {
    el.charCount.textContent = `${el.pasteBox.value.length} chars`;
  });

  el.pasteBox.addEventListener("keydown", (ev) => {
    const cmdEnter = (ev.metaKey || ev.ctrlKey) && ev.key === "Enter";
    if (cmdEnter) {
      ev.preventDefault();
      doFile(false);
    }
  });

  // Track which input was last focused so focus returns correctly after alt-tab
  let lastFocusedPane = "paste";
  el.askInput.addEventListener("focus", () => { lastFocusedPane = "ask"; });
  el.pasteBox.addEventListener("focus", () => { lastFocusedPane = "paste"; });

  window.addEventListener("focus", () => {
    if (lastFocusedPane === "ask") el.askInput.focus();
    else el.pasteBox.focus();
  });

  // ------------------------------------------------------------------
  // Date navigation
  // ------------------------------------------------------------------
  el.prevDay.addEventListener("click", () => loadDay(addDays(state.date, -1)));
  el.nextDay.addEventListener("click", () => loadDay(addDays(state.date, 1)));
  el.todayBtn.addEventListener("click", () => loadDay(todayStr()));
  el.datePicker.addEventListener("change", () => {
    if (storage_isValidDateClient(el.datePicker.value)) {
      loadDay(el.datePicker.value);
    }
  });

  function storage_isValidDateClient(v) {
    return /^\d{4}-\d{2}-\d{2}$/.test(v || "");
  }

  // ------------------------------------------------------------------
  // Tabs (mobile)
  // ------------------------------------------------------------------
  el.tabJournal.addEventListener("click", () => {
    el.tabJournal.classList.add("active");
    el.tabAsk.classList.remove("active");
    el.paneJournal.classList.remove("hidden");
    el.paneAsk.classList.add("hidden");
  });
  el.tabAsk.addEventListener("click", () => {
    el.tabAsk.classList.add("active");
    el.tabJournal.classList.remove("active");
    el.paneAsk.classList.remove("hidden");
    el.paneJournal.classList.add("hidden");
  });

  el.entriesBox.addEventListener("scroll", updateEntriesFade);

  // ------------------------------------------------------------------
  // Ask console
  // ------------------------------------------------------------------
  function renderAnswerBlock(query, answer) {
    const block = document.createElement("div");
    block.className = "qa-block";

    const qLine = document.createElement("div");
    qLine.className = "qa-query";
    qLine.textContent = `> ${query}`;
    block.appendChild(qLine);

    const answerDiv = document.createElement("div");
    answerDiv.className = "qa-answer";
    renderHeaderedText(answerDiv, answer);
    block.appendChild(answerDiv);

    const copyBtn = document.createElement("button");
    copyBtn.className = "qa-copy";
    copyBtn.textContent = "COPY";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(answer);
        copyBtn.textContent = "COPIED";
        copyBtn.classList.add("copied");
        setTimeout(() => {
          copyBtn.textContent = "COPY";
          copyBtn.classList.remove("copied");
        }, 1200);
      } catch (e) {
        copyBtn.textContent = "COPY FAILED";
      }
    });
    block.appendChild(copyBtn);

    el.askHistory.insertBefore(block, el.askHistory.firstChild);
  }

  function renderHeaderedText(container, text) {
    container.textContent = "";
    const headerRe = /^\*(MACRO THESIS|FLOWS SEEN|ECONOMICS|PRICE ACTION)\*$/;
    const lines = text.split("\n");
    for (const line of lines) {
      const m = line.match(headerRe);
      if (m) {
        const hdr = document.createElement("div");
        hdr.className = "hdr";
        hdr.textContent = line;
        container.appendChild(hdr);
      } else {
        container.appendChild(document.createTextNode(line));
      }
      container.appendChild(document.createTextNode("\n"));
    }
  }

  async function submitAsk() {
    const query = el.askInput.value.trim();
    if (!query) return;

    if (query.toLowerCase() === "eod") {
      renderEodConfirmCard();
      el.askInput.value = "";
      return;
    }

    el.askInput.value = "";
    const { status, body } = await apiAsk(state.date, query);
    if (status !== 200) {
      renderAnswerBlock(query, body.error || "Request failed.");
      return;
    }
    renderAnswerBlock(query, body.answer);
  }

  el.askInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      submitAsk();
    }
  });

  // ------------------------------------------------------------------
  // EOD inline confirm card
  // ------------------------------------------------------------------
  function renderEodConfirmCard() {
    // Remove any existing EOD confirm card to avoid duplicates
    const prev = el.askHistory.querySelector(".eod-block");
    if (prev) prev.remove();

    const count = state.day.entries.length;

    const block = document.createElement("div");
    block.className = "qa-block eod-block";

    const qLine = document.createElement("div");
    qLine.className = "qa-query";
    qLine.textContent = "> eod";
    block.appendChild(qLine);

    const card = document.createElement("div");
    card.className = "eod-card";

    const title = document.createElement("div");
    title.className = "eod-card-title";
    title.textContent = `Close EOD — ${state.date}`;
    card.appendChild(title);

    const body = document.createElement("div");
    body.className = "eod-card-body";
    const p1 = document.createElement("p");
    p1.textContent = `${count} ${count === 1 ? "entry" : "entries"} will be compiled.`;
    const p2 = document.createElement("p");
    p2.textContent = "compile → write → close → commit + push";
    body.appendChild(p1);
    body.appendChild(p2);
    card.appendChild(body);

    const actions = document.createElement("div");
    actions.className = "eod-card-actions";

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "ghost";
    cancelBtn.textContent = "CANCEL";
    cancelBtn.addEventListener("click", () => block.remove());

    const confirmBtn = document.createElement("button");
    confirmBtn.className = "file-btn";
    confirmBtn.textContent = "CONFIRM";
    confirmBtn.addEventListener("click", async () => {
      block.remove();
      const { status, body: rb } = await apiEod(state.date);
      if (status !== 200) {
        renderAnswerBlock("eod", rb.error || "EOD failed.");
        return;
      }
      renderAnswerBlock(
        "eod",
        `Closed ${state.date} — ${rb.count} entries — git: ${rb.git}\n\n${rb.report}`
      );
      await loadDay(state.date);
    });

    actions.appendChild(cancelBtn);
    actions.appendChild(confirmBtn);
    card.appendChild(actions);
    block.appendChild(card);

    el.askHistory.insertBefore(block, el.askHistory.firstChild);
  }

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------
  (async function boot() {
    const cfg = await apiGetConfig();
    state.mockMode = !!cfg.mockMode;
    el.mockBadge.classList.toggle("hidden", !state.mockMode);
    await loadDay(todayStr());
    el.pasteBox.focus();
  })();
})();

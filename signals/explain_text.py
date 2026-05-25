"""Generate plain-English summaries of explain data using a local Ollama model."""

import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import OLLAMA_URL, OLLAMA_MODEL

log = logging.getLogger(__name__)


def _build_prompt(sym: str, d: dict) -> str:
    signal = d.get("signal", "HOLD")
    score = d.get("ensemble_score") or 0.0
    xgb_pred = d.get("xgb_pred") or 0.0
    tft_pred = d.get("tft_pred") or 0.0
    snap = d.get("snapshot") or {}
    shap = (d.get("shap") or [])[:5]
    tft_interp = d.get("tft_interpretation")

    shap_lines = "\n".join(
        f"  - {s['feature']}: {'supports signal' if s['value'] > 0 else 'opposes signal'} ({s['value'] * 100:+.3f}%)"
        for s in shap
    )

    att_note = ""
    if tft_interp and tft_interp.get("attention"):
        att = tft_interp["attention"]
        peak_idx = max(range(len(att)), key=lambda i: att[i])
        days_ago = len(att) - 1 - peak_idx
        att_note = f"\n- TFT attention peaked {days_ago} days ago"

    snap_lines = []
    close = snap.get("close")
    rsi = snap.get("RSI_14")
    ret1d = snap.get("return_1d")
    ret5d = snap.get("return_5d")
    sentiment = snap.get("sentiment_mean")
    news_count = snap.get("sentiment_count")
    if close is not None:
        snap_lines.append(f"  - Price: ${close:.2f}")
    if rsi is not None:
        snap_lines.append(f"  - RSI 14: {rsi:.1f}")
    if ret1d is not None:
        snap_lines.append(f"  - 1d return: {ret1d * 100:+.2f}%")
    if ret5d is not None:
        snap_lines.append(f"  - 5d return: {ret5d * 100:+.2f}%")
    if sentiment is not None and news_count:
        snap_lines.append(f"  - News sentiment: {sentiment:.3f} ({int(news_count)} articles)")

    return (
        f"You are a quantitative trading analyst. Given the model output below, write exactly "
        f"2-3 plain English sentences explaining why {sym} received a {signal} signal. "
        f"Be specific — mention actual numbers from the data. "
        f"Do not use bullet points, headers, or any preamble.\n\n"
        f"Signal: {signal} (ensemble score {score:.0%})\n"
        f"XGBoost predicted return: {xgb_pred * 100:+.2f}%\n"
        f"TFT predicted return: {tft_pred * 100:+.2f}%\n\n"
        f"Top model drivers (SHAP):\n{shap_lines}\n\n"
        f"Market snapshot:\n" + "\n".join(snap_lines) + att_note
    )


def _call_ollama(sym: str, d: dict, model: str, timeout: int) -> tuple[str, str | None]:
    prompt = _build_prompt(sym, d)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return sym, text or None
    except requests.exceptions.ConnectionError:
        log.warning("Ollama not reachable at %s — skipping summaries", OLLAMA_URL)
        return sym, None
    except Exception as e:
        log.warning("Ollama summary failed for %s: %s", sym, e)
        return sym, None


def add_summaries(
    explain: dict[str, dict],
    model: str = OLLAMA_MODEL,
    max_workers: int = 4,
    timeout: int = 30,
) -> None:
    """Enrich each entry in explain with a plain-English 'summary' field (in-place).

    Silently skips if Ollama is not running — the rest of the pipeline is unaffected.
    """
    if not explain:
        return

    # Probe Ollama before spinning up the pool to avoid a cascade of connection errors.
    try:
        requests.get(OLLAMA_URL.rsplit("/", 1)[0], timeout=3)
    except requests.exceptions.ConnectionError:
        log.warning("Ollama not reachable — skipping plain-English summaries")
        return

    log.info(
        "Generating plain-English summaries for %d symbols via Ollama (%s, %d workers)",
        len(explain), model, max_workers,
    )
    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_call_ollama, sym, d, model, timeout): sym
            for sym, d in explain.items()
        }
        for future in as_completed(futures):
            sym, text = future.result()
            if text:
                explain[sym]["summary"] = text
                ok += 1

    log.info("Summaries generated: %d/%d", ok, len(explain))

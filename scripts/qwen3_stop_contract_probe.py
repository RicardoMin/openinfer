#!/usr/bin/env python3
"""Compare the Qwen3 typed stop contract with an un-migrated Qwen3.5 path.

The script talks to already-running OpenAI-compatible PegaInfer servers.  It
does not start a server and deliberately contains no model-specific launch
flags.  By default each target receives a full-vocabulary explicit stop set;
this guarantees that the first sampled token exercises the explicit-stop path.
Use --stop-token-id when a smaller, known stop set is preferred.

Every check validates the wire shape, not merely the presence of a field:

- `stop_reason` must be a numeric token ID from the requested stop set (or
  absent for a model-EOS finish);
- under a full-vocabulary stop set the first token always matches, so the
  explicit-stop runs must report exactly one completion token;
- the triggering token must carry a non-null numeric logprob;
- the exact trigger is verified by requesting `return_token_ids` and comparing
  the final returned token ID against `stop_reason`;
- when `--qwen3-eos-token-id` / `--qwen35-eos-token-id` is given, an EOS
  finish must end on exactly that trigger ID;
- the streaming case must terminate with `[DONE]` and must not emit content
  after the finish event;
- `/v1/models` must actually serve the requested model name.

Run `--self-check` to exercise the checks against a local mock service that
serves the malformed-response shapes these checks exist to reject; no GPU or
server binary is required. Both targets are reported side by side, and
`--require-legacy-gap` turns the expected adapted-passes/legacy-fails outcome
into an exit-code assertion.

Example:
    python3 scripts/qwen3_stop_contract_probe.py \
      --qwen3-url http://127.0.0.1:18081 --qwen3-model qwen3-adapted \
      --qwen35-url http://127.0.0.1:18082 --qwen35-model qwen35-legacy \
      --out stop-contract-ab.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PROMPT = "The capital of France is"
DEFAULT_QWEN3_URL = "http://127.0.0.1:18081"
DEFAULT_QWEN35_URL = "http://127.0.0.1:18082"
DEFAULT_QWEN3_MODEL = "qwen3-adapted"
DEFAULT_QWEN35_MODEL = "qwen35-legacy"
DEFAULT_QWEN3_VOCAB = 151_936
DEFAULT_QWEN35_VOCAB = 248_320


def endpoint(base_url: str, suffix: str) -> str:
    return base_url.rstrip("/") + suffix


def http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except HTTPError as error:
        raw = error.read()
        try:
            body: Any = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = raw.decode("utf-8", errors="replace")
        return {
            "http_status": error.code,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": body,
        }
    except (TimeoutError, URLError, OSError) as error:
        return {
            "http_status": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": str(error),
        }
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return {
            "http_status": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": f"invalid JSON response: {error}",
        }
    return {
        "http_status": status,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "body": body,
    }


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def token_ids_from_choice(choice: dict[str, Any]) -> list[int] | None:
    ids = choice.get("token_ids")
    if isinstance(ids, list) and all(is_int(item) for item in ids):
        return ids
    return None


def logprob_at_last(choice: dict[str, Any], token_count: int) -> float | None:
    """Finite logprob of the final emitted token, or None when absent.

    The logprob table must cover exactly the emitted tokens: a table shorter
    than the token sequence (a missing trigger entry) or a null final entry is
    a failure rather than an inherited value from an earlier token.
    """
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict) or token_count == 0:
        return None
    token_logprobs = logprobs.get("token_logprobs")
    if isinstance(token_logprobs, list):
        if len(token_logprobs) != token_count:
            return None
        value = token_logprobs[-1]
        return value if is_finite_number(value) else None
    content = logprobs.get("content")
    if isinstance(content, list):
        if len(content) != token_count:
            return None
        last = content[-1]
        if isinstance(last, dict):
            value = last.get("logprob")
            return value if is_finite_number(value) else None
    return None


def stream_token_count(choice: dict[str, Any]) -> int:
    """Number of tokens a streaming choice emits (token_ids, logprobs, or text)."""
    ids = token_ids_from_choice(choice)
    if ids is not None:
        return len(ids)
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        tokens = logprobs.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        content = logprobs.get("content")
        if isinstance(content, list):
            return len(content)
    return 0


def first_choice(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0]
    return choice if isinstance(choice, dict) else {}


def completion_summary(response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body")
    if not isinstance(body, dict):
        return {
            **response,
            "finish_reason": None,
            "stop_reason": None,
            "completion_tokens": None,
            "token_ids": None,
            "trigger_logprob": None,
        }
    choice = first_choice(body)
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    stop_reason = choice.get("stop_reason", body.get("stop_reason"))
    token_ids = token_ids_from_choice(choice)
    return {
        "http_status": response.get("http_status"),
        "elapsed_ms": response.get("elapsed_ms"),
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": stop_reason,
        "completion_tokens": usage.get("completion_tokens"),
        "token_ids": token_ids,
        "trigger_logprob": logprob_at_last(choice, len(token_ids) if token_ids is not None else 0),
    }


def completion_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool,
    stop_token_ids: list[int] | None,
    logprobs: int,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "stream": stream,
        "return_token_ids": True,
    }
    if stop_token_ids is not None:
        payload["stop_token_ids"] = stop_token_ids
    if logprobs > 0:
        payload["logprobs"] = logprobs
    return payload


def request_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    *,
    ignore_eos: bool,
    stop_token_ids: list[int] | None = None,
    logprobs: int = 0,
) -> dict[str, Any]:
    payload = completion_payload(model, prompt, max_tokens, ignore_eos, stop_token_ids, logprobs, stream=False)
    return completion_summary(http_json(endpoint(base_url, "/v1/completions"), payload, timeout))


def request_stream(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    *,
    ignore_eos: bool,
    stop_token_ids: list[int] | None = None,
    logprobs: int = 0,
) -> dict[str, Any]:
    payload = completion_payload(model, prompt, max_tokens, ignore_eos, stop_token_ids, logprobs, stream=True)
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        endpoint(base_url, "/v1/completions"),
        data=data,
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    events = 0
    emitted_tokens = 0
    stream_token_ids: list[int] = []
    last_content_logprob: float | None = None
    trigger_logprob_valid = False
    finish_reason: Any = None
    stop_reason: Any = None
    done_seen = False
    content_after_finish = False
    malformed_chunks = 0
    finished = False
    status = None
    error = None
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_line = line[5:].strip()
                if data_line == "[DONE]":
                    done_seen = True
                    break
                try:
                    chunk = json.loads(data_line)
                except json.JSONDecodeError:
                    malformed_chunks += 1
                    continue
                if not isinstance(chunk, dict):
                    malformed_chunks += 1
                    continue
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    # A usage-only frame carries no content and is legal.
                    if isinstance(chunk.get("usage"), dict):
                        continue
                    malformed_chunks += 1
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                text = choice.get("text")
                frame_ids = token_ids_from_choice(choice)
                token_count = len(frame_ids) if frame_ids is not None else stream_token_count(choice)
                has_content = bool(text) or token_count > 0
                this_finish = choice.get("finish_reason")
                this_stop = chunk.get("stop_reason", choice.get("stop_reason"))
                terminal_event = this_finish is not None or this_stop is not None
                if finished:
                    if has_content or terminal_event:
                        content_after_finish = True
                    continue
                if has_content:
                    # Count content even when it shares a frame with the finish
                    # metadata; a terminal frame must not hide emitted tokens.
                    events += 1
                    emitted_tokens += token_count if token_count else 1
                    if frame_ids is not None:
                        stream_token_ids.extend(frame_ids)
                    value = logprob_at_last(choice, token_count)
                    trigger_logprob_valid = value is not None
                    if value is not None:
                        last_content_logprob = value
                if terminal_event:
                    if this_finish is not None:
                        finish_reason = this_finish
                    if stop_reason is None and this_stop is not None:
                        stop_reason = this_stop
                    finished = True
    except HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")
    except (TimeoutError, URLError, OSError) as exc:
        error = str(exc)
    return {
        "http_status": status,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "stream_events": events,
        "emitted_tokens": emitted_tokens,
        "token_ids": stream_token_ids,
        "trigger_logprob": last_content_logprob if trigger_logprob_valid else None,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "completion_tokens": None,
        "done_seen": done_seen,
        "content_after_finish": content_after_finish,
        "malformed_chunks": malformed_chunks,
        "error": error,
    }


def probe_models(base_url: str, timeout: float, expected_model: str) -> dict[str, Any]:
    result = http_json(endpoint(base_url, "/v1/models"), None, timeout)
    body = result.get("body")
    ids: list[str] = []
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        ids = [
            item.get("id")
            for item in body["data"]
            if isinstance(item, dict) and item.get("id")
        ]
    result["model_ids"] = ids
    result["expected_model_present"] = expected_model in ids
    return result


def is_int(value: Any) -> bool:
    # JSON `true` parses to Python True, which is also an `int` subclass;
    # exclude it so a boolean wire value can never satisfy a numeric check.
    return isinstance(value, int) and not isinstance(value, bool)


def numeric_stop_reason(result: dict[str, Any]) -> bool:
    return is_int(result.get("stop_reason"))


def is_stop_finish(result: dict[str, Any]) -> bool:
    return result.get("http_status") == 200 and result.get("finish_reason") == "stop"


def typed_stop(result: dict[str, Any], stop_ids: list[int], full_vocab: bool) -> bool:
    """A typed request stop: the reported stop_reason is the emitted trigger."""
    if not is_stop_finish(result):
        return False
    if not numeric_stop_reason(result) or result["stop_reason"] not in stop_ids:
        return False
    ids = result.get("token_ids")
    if not isinstance(ids, list) or not ids:
        return False
    tokens = result.get("completion_tokens")
    if not is_int(tokens) or tokens != len(ids) or tokens < 1:
        return False
    if ids[-1] != result["stop_reason"]:
        return False
    if full_vocab:
        # The first sampled token is in the full-vocabulary stop set, so the
        # trigger must be the only completion token.
        return len(ids) == 1
    return True


def eos_or_typed_stop(
    result: dict[str, Any],
    stop_ids: list[int],
    full_vocab: bool,
    eos_id: int | None,
) -> bool:
    """EOS-enabled explicit-stop case: a model EOS may win over the stop set."""
    if not is_stop_finish(result):
        return False
    ids = result.get("token_ids")
    if not isinstance(ids, list) or not ids:
        return False
    tokens = result.get("completion_tokens")
    if not is_int(tokens) or tokens != len(ids) or tokens < 1:
        return False
    stop_reason = result.get("stop_reason")
    if stop_reason is None:
        # A model EOS won. With a configured EOS ID the final token must be
        # exactly that trigger; otherwise only the count and terminal shape
        # are verifiable.
        if eos_id is not None and ids[-1] != eos_id:
            return False
        return not full_vocab or len(ids) == 1
    if not is_int(stop_reason) or stop_reason not in stop_ids:
        return False
    if ids[-1] != stop_reason:
        return False
    return not full_vocab or len(ids) == 1


def length_control(result: dict[str, Any], max_tokens: int) -> bool:
    ids = result.get("token_ids")
    return (
        result.get("http_status") == 200
        and result.get("finish_reason") == "length"
        and result.get("stop_reason") is None
        and isinstance(ids, list)
        and len(ids) == max_tokens
        and is_int(result.get("completion_tokens"))
        and result.get("completion_tokens") == max_tokens
    )


def build_stop_ids(vocab_size: int, selected_id: int | None) -> list[int]:
    if selected_id is not None:
        if selected_id < 0 or selected_id >= vocab_size:
            raise ValueError(f"--stop-token-id must be in [0, {vocab_size}), got {selected_id}")
        return [selected_id]
    return list(range(vocab_size))


def run_target(
    name: str,
    url: str,
    model: str,
    vocab_size: int,
    args: argparse.Namespace,
    eos_token_id: int | None = None,
) -> dict[str, Any]:
    stop_ids = build_stop_ids(vocab_size, args.stop_token_id)
    full_vocab = args.stop_token_id is None
    model_probe = probe_models(url, args.timeout, model)

    def call(ignore_eos: bool, ids: list[int] | None, lp: int = 0) -> dict[str, Any]:
        return request_completion(
            url,
            model,
            args.prompt,
            args.max_tokens,
            args.timeout,
            ignore_eos=ignore_eos,
            stop_token_ids=ids,
            logprobs=lp,
        )

    cases: dict[str, Any] = {}
    cases["control"] = call(True, None)
    cases["explicit_stop_ignore_eos"] = call(True, stop_ids)
    cases["explicit_stop_eos_enabled"] = call(False, stop_ids)
    cases["stop_ascending"] = call(True, stop_ids)
    cases["stop_descending"] = call(True, list(reversed(stop_ids)))
    cases["trigger_logprob"] = call(True, stop_ids, lp=1)
    cases["streaming"] = request_stream(
        url,
        model,
        args.prompt,
        args.max_tokens,
        args.timeout,
        ignore_eos=True,
        stop_token_ids=stop_ids,
        logprobs=1,
    )

    mixed_jobs: list[tuple[str, bool]] = [("control", False)] * 3 + [("explicit", True)] * 3

    def mixed_call(job: tuple[str, bool]) -> dict[str, Any]:
        _, explicit = job
        return request_completion(
            url,
            model,
            args.prompt,
            args.max_tokens,
            args.timeout,
            ignore_eos=True,
            stop_token_ids=stop_ids if explicit else None,
        )

    with ThreadPoolExecutor(max_workers=len(mixed_jobs)) as pool:
        mixed_results = list(pool.map(mixed_call, mixed_jobs))
    cases["mixed"] = [
        {"kind": kind, "result": result}
        for (kind, _), result in zip(mixed_jobs, mixed_results)
    ]

    mixed_controls = [item["result"] for item in cases["mixed"] if item["kind"] == "control"]
    mixed_explicit = [item["result"] for item in cases["mixed"] if item["kind"] == "explicit"]
    streaming = cases["streaming"]
    checks = {
        "model_present": bool(model_probe.get("expected_model_present")),
        "baseline_control": length_control(cases["control"], args.max_tokens),
        "explicit_stop_ignore_eos": typed_stop(cases["explicit_stop_ignore_eos"], stop_ids, full_vocab),
        "explicit_stop_eos_enabled": eos_or_typed_stop(
            cases["explicit_stop_eos_enabled"], stop_ids, full_vocab, eos_token_id
        ),
        "stop_set_order_invariant": (
            typed_stop(cases["stop_ascending"], stop_ids, full_vocab)
            and typed_stop(cases["stop_descending"], stop_ids, full_vocab)
            and cases["stop_ascending"].get("stop_reason")
            == cases["stop_descending"].get("stop_reason")
            and cases["stop_ascending"].get("completion_tokens")
            == cases["stop_descending"].get("completion_tokens")
        ),
        "trigger_logprob_preserved": (
            typed_stop(cases["trigger_logprob"], stop_ids, full_vocab)
            and is_finite_number(cases["trigger_logprob"].get("trigger_logprob"))
        ),
        "stream_reports_typed_stop": (
            streaming.get("http_status") == 200
            and streaming.get("finish_reason") == "stop"
            and numeric_stop_reason(streaming)
            and streaming.get("stop_reason") in stop_ids
            and streaming.get("stream_events", 0) > 0
            and streaming.get("done_seen") is True
            and not streaming.get("content_after_finish")
            and streaming.get("malformed_chunks", 0) == 0
            and is_finite_number(streaming.get("trigger_logprob"))
            and isinstance(streaming.get("token_ids"), list)
            and len(streaming["token_ids"]) > 0
            and streaming["token_ids"][-1] == streaming.get("stop_reason")
            and streaming.get("emitted_tokens") == len(streaming["token_ids"])
            and (
                streaming.get("emitted_tokens") == 1
                if full_vocab
                else (is_int(streaming.get("emitted_tokens")) and streaming["emitted_tokens"] >= 1)
            )
        ),
        "mixed_controls_pass_3_of_3": sum(length_control(item, args.max_tokens) for item in mixed_controls) == 3,
        "mixed_explicit_stops_pass_3_of_3": sum(typed_stop(item, stop_ids, full_vocab) for item in mixed_explicit) == 3,
    }
    return {
        "name": name,
        "url": url,
        "model": model,
        "vocab_size": vocab_size,
        "stop_set": "single" if args.stop_token_id is not None else "full-vocabulary",
        "stop_set_size": len(stop_ids),
        "model_probe": model_probe,
        "cases": cases,
        "checks": checks,
        "new_contract_passed": all(checks.values()),
    }


def print_target(target: dict[str, Any]) -> None:
    print(f"\n{target['name']} ({target['model']})")
    print("case                         finish  stop_reason  tokens  http")
    for name, result in target["cases"].items():
        if name == "mixed":
            continue
        if name == "streaming":
            print(
                f"{name:28} {str(result.get('finish_reason')):7} "
                f"{str(result.get('stop_reason')):12} {'n/a':>7} "
                f"{str(result.get('http_status'))}  events={result.get('stream_events')} "
                f"done={result.get('done_seen')} tail={result.get('content_after_finish')}"
            )
            continue
        print(
            f"{name:28} {str(result.get('finish_reason')):7} "
            f"{str(result.get('stop_reason')):12} {str(result.get('completion_tokens')):>7} "
            f"{str(result.get('http_status'))}"
        )
    print("checks:")
    for name, passed in target["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"overall new contract: {'PASS' if target['new_contract_passed'] else 'FAIL'}")


def print_comparison(qwen3: dict[str, Any], qwen35: dict[str, Any]) -> None:
    print("\ncontract checks (adapted vs legacy):")
    print(f"{'check':28} {'qwen3':7} {'qwen35':7}")
    for name in qwen3["checks"]:
        left = "PASS" if qwen3["checks"][name] else "FAIL"
        right = "PASS" if qwen35["checks"][name] else "FAIL"
        print(f"{name:28} {left:7} {right:7}")
    print(
        f"{'overall new contract':28} "
        f"{'PASS' if qwen3['new_contract_passed'] else 'FAIL':7} "
        f"{'PASS' if qwen35['new_contract_passed'] else 'FAIL':7}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen3-url", default=DEFAULT_QWEN3_URL)
    parser.add_argument("--qwen3-model", default=DEFAULT_QWEN3_MODEL)
    parser.add_argument("--qwen3-vocab-size", type=int, default=DEFAULT_QWEN3_VOCAB)
    parser.add_argument("--qwen35-url", default=DEFAULT_QWEN35_URL)
    parser.add_argument("--qwen35-model", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--qwen35-vocab-size", type=int, default=DEFAULT_QWEN35_VOCAB)
    parser.add_argument(
        "--qwen3-eos-token-id",
        type=int,
        default=None,
        help="Verify an EOS finish ends on exactly this token ID (optional)",
    )
    parser.add_argument(
        "--qwen35-eos-token-id",
        type=int,
        default=None,
        help="Verify an EOS finish ends on exactly this token ID (optional)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--stop-token-id",
        type=int,
        help="Use one explicit stop ID instead of the default full-vocabulary set",
    )
    parser.add_argument("--out", type=Path, help="Write the complete result as JSON")
    parser.add_argument(
        "--strict-both",
        action="store_true",
        help="Return failure unless both targets satisfy every new-contract check",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run the checks against a local mock service that serves malformed "
        "responses; verify each is rejected. Requires no server or GPU.",
    )
    parser.add_argument(
        "--require-legacy-gap",
        action="store_true",
        help="Return failure unless the adapted target passes every check and the "
        "legacy target fails at least one (the expected A/B outcome).",
    )
    return parser.parse_args()


SELF_CHECK_MODEL = "qwen3-adapted"
SELF_CHECK_STOP_ID = 12095
SELF_CHECK_EOS_ID = 151645

_MODE = "valid"
_MODE_LOCK = threading.Lock()


def set_mock_mode(mode: str) -> None:
    global _MODE
    with _MODE_LOCK:
        _MODE = mode


def mock_mode() -> str:
    with _MODE_LOCK:
        return _MODE


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/v1/models"):
            if mock_mode() == "wrong_model":
                self._json({"data": [{"id": "different-model"}]})
            else:
                self._json({"data": [{"id": SELF_CHECK_MODEL}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return
        if not self.path.rstrip("/").endswith("/v1/completions"):
            self.send_error(404)
            return
        mode = mock_mode()
        if body.get("stream"):
            self._stream(mode)
        else:
            self._completion(body, mode)

    def _completion(self, body: dict[str, Any], mode: str) -> None:
        explicit = body.get("stop_token_ids") is not None
        max_tokens = body.get("max_tokens", 8)
        if not explicit:
            control_choice: dict[str, Any] = {
                "text": " word",
                "index": 0,
                "finish_reason": "length",
                "logprobs": None,
            }
            if mode != "missing_token_ids":
                control_choice["token_ids"] = list(range(max_tokens))
            self._json(
                {
                    "choices": [control_choice],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": max_tokens,
                        "total_tokens": 5 + max_tokens,
                    },
                }
            )
            return
        if mode == "eos_win" and not body.get("ignore_eos"):
            self._json(
                {
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "finish_reason": "stop",
                            "logprobs": None,
                            "stop_reason": None,
                            "token_ids": [SELF_CHECK_EOS_ID],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                }
            )
            return
        stop_reason: Any = SELF_CHECK_STOP_ID if mode != "string_stop_reason" else "oops"
        tokens = max_tokens if mode == "extra_tokens" else 1
        token_ids = list(range(tokens)) if mode == "extra_tokens" else [SELF_CHECK_STOP_ID]
        if mode == "missing_trigger_logprob":
            tokens = 2
            token_ids = [999, SELF_CHECK_STOP_ID]
        if mode == "wrong_trigger_id":
            stop_reason = 17
            token_ids = [SELF_CHECK_STOP_ID]
        choice: dict[str, Any] = {
            "text": " stop",
            "index": 0,
            "finish_reason": "stop",
            "logprobs": None,
        }
        if mode != "missing_token_ids":
            choice["token_ids"] = token_ids
        if (body.get("logprobs") or 0) > 0:
            if mode == "null_logprobs":
                choice["logprobs"] = {"content": [{"token": "<trigger>", "logprob": None}]}
            elif mode == "missing_trigger_logprob":
                # One logprob entry short of the two emitted tokens.
                choice["logprobs"] = {"tokens": ["<prev>"], "token_logprobs": [-0.5]}
            else:
                choice["logprobs"] = {"content": [{"token": "<trigger>", "logprob": -0.53125}]}
        choice["stop_reason"] = stop_reason
        self._json(
            {
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": tokens,
                    "total_tokens": 5 + tokens,
                },
            }
        )

    def _stream(self, mode: str) -> None:
        stop_reason: Any = SELF_CHECK_STOP_ID if mode != "string_stop_reason" else "oops"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(chunk: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

        first_choice: dict[str, Any] = {
            "text": "",
            "index": 0,
            "finish_reason": None,
            "token_ids": [SELF_CHECK_STOP_ID],
        }
        if mode != "stream_missing_trigger_logprob":
            first_choice["logprobs"] = {
                "tokens": [" stop"],
                "token_logprobs": [-0.53125],
                "top_logprobs": [],
            }
        emit({"id": "cmpl-1", "choices": [first_choice]})
        if mode in ("terminal_frame_extra_token", "terminal_frame_extra_token_full_vocab"):
            # The finish frame smuggles a second token next to the metadata.
            smuggled = [17] if mode == "terminal_frame_extra_token" else [SELF_CHECK_STOP_ID]
            emit(
                {
                    "id": "cmpl-1",
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "finish_reason": "stop",
                            "logprobs": None,
                            "token_ids": smuggled,
                        }
                    ],
                    "stop_reason": stop_reason,
                }
            )
        else:
            emit(
                {
                    "id": "cmpl-1",
                    "choices": [{"text": "", "index": 0, "finish_reason": "stop", "logprobs": None}],
                    "stop_reason": stop_reason,
                }
            )
        if mode == "stream_tail":
            emit(
                {
                    "id": "cmpl-1",
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": {
                                "tokens": [" tail"],
                                "token_logprobs": [-1.0],
                                "top_logprobs": [],
                            },
                            "token_ids": [17],
                        }
                    ],
                }
            )
        if mode != "stream_missing_done":
            self.wfile.write(b"data: [DONE]\n\n")


def run_self_check() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"

    class Args:
        prompt = DEFAULT_PROMPT
        max_tokens = 8
        timeout = 10.0
        stop_token_id = None

    cases = [
        ("valid", None, None, True, "valid service passes every check"),
        (
            "eos_win",
            SELF_CHECK_STOP_ID,
            SELF_CHECK_EOS_ID,
            True,
            "an EOS finish must end on the configured EOS trigger",
        ),
        ("string_stop_reason", SELF_CHECK_STOP_ID, None, False, "string stop_reason must be rejected"),
        ("null_logprobs", SELF_CHECK_STOP_ID, None, False, "null trigger logprob must be rejected"),
        ("stream_tail", SELF_CHECK_STOP_ID, None, False, "content after the finish event must be rejected"),
        ("wrong_model", SELF_CHECK_STOP_ID, None, False, "model-name mismatch must be rejected"),
        (
            "extra_tokens",
            None,
            None,
            False,
            "extra completion tokens under a full-vocabulary stop set must be rejected",
        ),
        (
            "missing_trigger_logprob",
            SELF_CHECK_STOP_ID,
            None,
            False,
            "a missing final logprob must not inherit the previous token's value",
        ),
        (
            "wrong_trigger_id",
            None,
            None,
            False,
            "stop_reason must match the actual final token ID",
        ),
        (
            "terminal_frame_extra_token",
            SELF_CHECK_STOP_ID,
            None,
            False,
            "tokens sharing a frame with finish metadata must be counted",
        ),
        (
            "terminal_frame_extra_token_full_vocab",
            None,
            None,
            False,
            "a duplicated trigger in the finish frame must still be counted",
        ),
        (
            "missing_token_ids",
            None,
            None,
            False,
            "responses without token IDs cannot satisfy the trigger checks",
        ),
        (
            "stream_missing_done",
            SELF_CHECK_STOP_ID,
            None,
            False,
            "a stream that never sends [DONE] must be rejected",
        ),
        (
            "stream_missing_trigger_logprob",
            SELF_CHECK_STOP_ID,
            None,
            False,
            "the trigger frame must carry its own finite logprob",
        ),
    ]
    failures = 0
    for mode, stop_id, eos_id, expect_pass, label in cases:
        set_mock_mode(mode)
        args = Args()
        args.stop_token_id = stop_id
        target = run_target("qwen3_adapted", base_url, SELF_CHECK_MODEL, DEFAULT_QWEN3_VOCAB, args, eos_id)
        passed = target["new_contract_passed"]
        ok = passed == expect_pass
        if not ok:
            failures += 1
            for check_name, check_value in target["checks"].items():
                if not check_value:
                    print(f"    unexpected check state: {check_name}=FAIL")
        print(
            f"[{'PASS' if ok else 'FAIL'}] self-check {mode}: "
            f"overall={'PASS' if passed else 'FAIL'}, expected={'PASS' if expect_pass else 'FAIL'} ({label})"
        )
    server.shutdown()
    thread.join(timeout=5.0)
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.self_check:
        return run_self_check()
    if args.max_tokens <= 0:
        print("--max-tokens must be positive", file=sys.stderr)
        return 2
    try:
        qwen3 = run_target(
            "qwen3_adapted",
            args.qwen3_url,
            args.qwen3_model,
            args.qwen3_vocab_size,
            args,
            args.qwen3_eos_token_id,
        )
        qwen35 = run_target(
            "qwen35_legacy",
            args.qwen35_url,
            args.qwen35_model,
            args.qwen35_vocab_size,
            args,
            args.qwen35_eos_token_id,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    comparison = {
        "qwen3_new_contract_passed": qwen3["new_contract_passed"],
        "qwen35_new_contract_passed": qwen35["new_contract_passed"],
        "legacy_gap_observed": (
            qwen3["new_contract_passed"] and not qwen35["new_contract_passed"]
        ),
    }
    report = {
        "schema_version": 2,
        "config": {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "stop_mode": "single" if args.stop_token_id is not None else "full-vocabulary",
            "stop_token_id": args.stop_token_id,
            "qwen3_eos_token_id": args.qwen3_eos_token_id,
            "qwen35_eos_token_id": args.qwen35_eos_token_id,
        },
        "targets": {"qwen3_adapted": qwen3, "qwen35_legacy": qwen35},
        "comparison": comparison,
    }
    print_target(qwen3)
    print_target(qwen35)
    print_comparison(qwen3, qwen35)
    print("\ncomparison:")
    print(json.dumps(comparison, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    if not qwen3["new_contract_passed"]:
        return 1
    if args.strict_both and not qwen35["new_contract_passed"]:
        return 1
    if args.require_legacy_gap and not comparison["legacy_gap_observed"]:
        print("expected adapted-passes/legacy-fails gap was not observed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os

import engine


def _capture_command(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, command, *args, **kwargs):
            captured["command"] = command
            self.pid = 1234

    monkeypatch.setattr(engine.subprocess, "Popen", FakePopen)
    return captured


def test_extra_args_appended_last(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "dummy")
    monkeypatch.setenv(
        "SGLANG_EXTRA_ARGS", "--mem-fraction-static 0.85 --enable-metrics"
    )

    captured = _capture_command(monkeypatch)

    eng = engine.SGlangEngine(model="dummy", host="0.0.0.0", port=30000)
    eng.start_server()

    command = captured["command"]
    extra_tokens = ["--mem-fraction-static", "0.85", "--enable-metrics"]

    # Extra tokens are the consecutive trailing tokens of the command.
    assert command[-len(extra_tokens):] == extra_tokens

    # And they appear after the model-path option (an earlier-built token).
    assert command.index("--model-path") < command.index("--mem-fraction-static")


def test_no_extra_args_when_unset(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "dummy")
    monkeypatch.delenv("SGLANG_EXTRA_ARGS", raising=False)

    captured = _capture_command(monkeypatch)

    eng = engine.SGlangEngine(model="dummy", host="0.0.0.0", port=30000)
    eng.start_server()

    command = captured["command"]
    assert "--mem-fraction-static" not in command
    assert "--enable-metrics" not in command
    # Command ends with the model-path option pair (no trailing extras).
    assert command[-2:] == ["--model-path", "dummy"]

import io

from sim2real.console_output import compact_deployment_console


def test_compact_console_keeps_stages_warnings_and_failures(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    with compact_deployment_console():
        print("[RealSense] verbose camera details")
        print("[Automatic reset PASS] both devices stopped")
        print("[Throw trigger ARMED] throw now")
        print("[Throw trigger DETECTED] frame=42")
        print("[Object grounding SEARCHING] hold target still")
        print("[Object grounding RETRY] far candidate rejected")
        print("[Object grounding TRACKED] target accepted")
        print("[Perception UI READY] windows opened")
        print("[Provider][WARN] transient issue", file=__import__("sys").stderr)
        print("V94 deployment: FAILED: example", file=__import__("sys").stderr)

    assert "RealSense" not in stdout.getvalue()
    assert "Automatic reset PASS" in stdout.getvalue()
    assert "Throw trigger ARMED" in stdout.getvalue()
    assert "Throw trigger DETECTED" in stdout.getvalue()
    assert "Object grounding SEARCHING" in stdout.getvalue()
    assert "Object grounding RETRY" in stdout.getvalue()
    assert "Object grounding TRACKED" in stdout.getvalue()
    assert "Perception UI READY" in stdout.getvalue()
    assert "WARN" in stderr.getvalue()
    assert "FAILED" in stderr.getvalue()


def test_compact_console_can_be_disabled(monkeypatch):
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)

    with compact_deployment_console(enabled=False):
        print("full diagnostic")

    assert stdout.getvalue() == "full diagnostic\n"

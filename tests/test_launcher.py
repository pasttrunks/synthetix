import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_launcher():
    spec = importlib.util.spec_from_file_location("launch_synthetix", ROOT / "launch_synthetix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_resolves_repo_root():
    mod = load_launcher()
    assert mod.REPO_ROOT == ROOT
    assert mod.SERVER_SCRIPT == ROOT / "server.py"


def test_backend_labels_match_ui_wording():
    mod = load_launcher()
    assert mod.backend_label("desklib_academic") == (
        "Academic Sensitive — high recall, high false-positive risk"
    )
    assert mod.backend_label("hc3_roberta") == (
        "Fast Baseline — faster, but frequently misses unfamiliar AI writing"
    )
    assert mod.backend_label("balanced_review") == (
        "Experimental Comparison — two detectors with an uncertain outcome when they disagree"
    )


def test_health_url():
    mod = load_launcher()
    assert mod.health_url(8000) == "http://127.0.0.1:8000/health"


def test_windows_launchers_pass_correct_backend():
    start = (ROOT / "Start Synthetix.cmd").read_text(encoding="utf-8")
    fast = (ROOT / "Start Synthetix Fast.cmd").read_text(encoding="utf-8")
    balanced = (ROOT / "Start Synthetix Balanced.cmd").read_text(encoding="utf-8")
    assert "--backend desklib_academic" in start
    assert "--backend hc3_roberta" in fast
    assert "--backend balanced_review" in balanced
    assert "launch_synthetix.py" in start
    assert "launch_synthetix.py" in fast
    assert "launch_synthetix.py" in balanced


def test_windows_launchers_find_python_without_path():
    """Launchers must fall back to common Python install locations when py/python
    are not on PATH (the environment that broke the original .cmd files)."""
    for name in ("Start Synthetix.cmd", "Start Synthetix Fast.cmd", "Start Synthetix Balanced.cmd"):
        content = (ROOT / name).read_text(encoding="utf-8")
        assert "%LocalAppData%\\Programs\\Python\\Python312\\python.exe" in content
        assert 'where py >nul 2>nul' in content
        assert 'ERROR: Python 3.10 or newer was not found.' in content


def test_ui_backend_banner_notices_and_states():
    html = (ROOT / "ai_detector.html").read_text(encoding="utf-8")
    expected = [
        "Experimental writing signal — not a probability or proof of AI authorship.",
        "Loading detector…",
        "Detector ready",
        "Analyzing locally…",
        "This mode is intentionally sensitive and may flag human writing. Review the sentence evidence rather than relying on the overall score.",
        "This mode may fail to identify AI-generated text outside its training distribution.",
        "This mode runs two detectors. When they disagree, the result is uncertain and no classification is made.",
        "Experimental Comparison — two detectors with an uncertain outcome when they disagree",
        "Strong agreement",
        "Low agreement",
        "Uncertain disagreement",
        "The detectors disagree. No reliable classification can be made from this analysis.",
        "Upload Document",
        "Cancel",
        "backendBanner",
        "modeNotice",
        "errorBox",
        "resultBackend",
    ]
    for text in expected:
        assert text in html, f"missing in ai_detector.html: {text}"


def test_ui_does_not_overclaim():
    html = (ROOT / "ai_detector.html").read_text(encoding="utf-8")
    for bad in ("100% accurate", "Verified human", "production ready", "Detected GPT-4"):
        assert bad.lower() not in html.lower(), f"forbidden claim in ai_detector.html: {bad}"

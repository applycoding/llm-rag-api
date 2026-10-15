from app.evaluate import format_report, main, run_evaluation


def test_sample_evaluation_passes() -> None:
    report = run_evaluation()
    assert report.passed, format_report(report)
    assert len(report.results) == 5


def test_cli_exits_zero(capsys) -> None:
    assert main() == 0
    captured = capsys.readouterr()
    assert "5/5 cases passed" in captured.out

from chatbot.app import EMPTY_TOTALS, format_stats, respond


def test_format_stats():
    s = format_stats({"input_tokens": 1234, "output_tokens": 56, "cost_usd": 0.01234})
    assert "1,234" in s and "56" in s and "$0.0123" in s


async def test_respond_accumulates_totals(monkeypatch):
    from chatbot import app
    from chatbot.llm import stream_completion as real_stream

    def fake_stream(messages, model=None, temperature=None, **kw):
        return real_stream(messages, model="gpt-4o", mock_response="Hi!")

    monkeypatch.setattr(app, "stream_completion", fake_stream)
    outs = [o async for o in respond("hello", [], "default", "gpt-4o", 0.7, dict(EMPTY_TOTALS))]
    text, stats, totals = outs[-1]
    assert text == "Hi!"
    assert totals["input_tokens"] > 0 and totals["output_tokens"] > 0
    assert "Input tokens" in stats
    # second turn adds to the running totals
    outs2 = [o async for o in respond("again", [], "default", "gpt-4o", 0.7, totals)]
    assert outs2[-1][2]["output_tokens"] > totals["output_tokens"]

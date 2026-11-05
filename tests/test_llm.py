from chatbot.llm import stream_completion


async def test_stream_completion_yields_deltas_with_mock_provider():
    messages = [{"role": "user", "content": "hello"}]
    stream = stream_completion(messages, model="gpt-4o", mock_response="Hello there!")
    chunks = [d async for d in stream]
    assert chunks
    assert "".join(chunks) == "Hello there!"


async def test_stream_completion_reports_usage_after_iteration():
    messages = [{"role": "user", "content": "hello"}]
    stream = stream_completion(messages, model="gpt-4o", mock_response="Hello there!")
    assert stream.usage is None
    async for _ in stream:
        pass
    assert stream.usage is not None
    assert stream.usage.input_tokens > 0
    assert stream.usage.output_tokens > 0
    assert stream.usage.cost_usd >= 0.0

from chatbot.llm import stream_completion


async def test_stream_completion_yields_deltas_with_mock_provider():
    messages = [{"role": "user", "content": "hello"}]
    chunks = [
        d async for d in stream_completion(messages, model="gpt-4o", mock_response="Hello there!")
    ]
    assert chunks
    assert "".join(chunks) == "Hello there!"

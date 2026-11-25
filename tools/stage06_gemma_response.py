"""Gemma 4 transport decoding for the original-model text-only pilot."""

NATIVE_RESPONSE_VERSION = "gemma4_native_v1"
TURN_ENDINGS = {1: "<eos>", 106: "<turn|>"}
THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "<channel|>"


class ResponseEnvelopeError(ValueError):
    """A transport error whose reason code is safe to record."""


def extract_assistant_content(processor, raw, *, prefix, stop_token_id):
    """Use the checkpoint parser and verify that it preserves the answer text."""
    expected = raw
    if stop_token_id is not None:
        ending = TURN_ENDINGS.get(stop_token_id)
        if ending is None or not raw.endswith(ending):
            raise ResponseEnvelopeError("unexpected_terminal_token")
        expected = expected.removesuffix(ending)

    if expected.startswith(THOUGHT_OPEN):
        _, close, expected = expected[len(THOUGHT_OPEN):].partition(THOUGHT_CLOSE)
        if not close:
            raise ResponseEnvelopeError("unclosed_thought_channel")

    # The native text parser strips whitespace by default. Keep the exact content
    # so that even non-JSON whitespace is checked by the strict JSON parser.
    template = processor.tokenizer.response_template
    fields = template["fields"]
    content_field = fields["content"]
    template = {
        **template,
        "fields": {
            **fields,
            "content": {
                **content_field,
                "content_args": {**content_field.get("content_args", {}), "strip": False},
            },
        },
    }
    message = processor.parse_response(raw, schema=template, prefix=prefix)
    if message.get("role") != "assistant" or message.keys() - {"role", "content", "thinking"}:
        raise ResponseEnvelopeError("unexpected_assistant_message")
    content = message.get("content", "")
    # A native parser can replace an earlier content segment with a later one.
    # Fail if it discarded anything outside the leading thought or verified EOS.
    if content != expected:
        raise ResponseEnvelopeError("assistant_content_changed")
    return content

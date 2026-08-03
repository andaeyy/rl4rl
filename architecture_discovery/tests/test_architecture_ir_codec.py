import json

import pytest

from architecture_ir.codec import (
    IRDecodeError,
    MAX_IR_JSON_BYTES,
    decode_graph_json,
    encode_graph_json,
)
from tests.test_architecture_ir_graph import valid_graph


def test_strict_ir_json_round_trip_is_canonical():
    graph = valid_graph(metadata={"parameter_count": 10**9})
    encoded = encode_graph_json(graph)
    decoded = decode_graph_json(encoded)
    assert decoded.graph_hash == graph.graph_hash
    assert encode_graph_json(decoded) == encoded


def test_ir_decoder_rejects_duplicate_and_unknown_fields():
    with pytest.raises(IRDecodeError, match="duplicate JSON object key"):
        decode_graph_json('{"graph_id":"first","graph_id":"second"}')

    payload = valid_graph().to_dict()
    payload["candidate_python"] = "exec('unsafe')"
    with pytest.raises(IRDecodeError, match="unknown keys"):
        decode_graph_json(json.dumps(payload))


def test_ir_decoder_rejects_nonfinite_numbers_and_excessive_input():
    payload = valid_graph().to_dict()
    payload["metadata"]["unstable"] = float("nan")
    with pytest.raises(IRDecodeError, match="non-finite"):
        decode_graph_json(json.dumps(payload))

    with pytest.raises(IRDecodeError, match="byte input limit"):
        decode_graph_json(" " * (MAX_IR_JSON_BYTES + 1))


def test_ir_decoder_rejects_bad_enums_and_port_types():
    payload = valid_graph().to_dict()
    payload["nodes"][0]["kind"] = "python_callback"
    with pytest.raises(IRDecodeError, match=r"nodes\[0\]"):
        decode_graph_json(json.dumps(payload))

    payload = valid_graph().to_dict()
    payload["edges"][0]["target_port"] = 0.5
    with pytest.raises(IRDecodeError, match="nonnegative integer"):
        decode_graph_json(json.dumps(payload))

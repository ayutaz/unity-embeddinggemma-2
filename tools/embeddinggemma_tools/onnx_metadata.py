"""Inspect ONNX operator names and small constants without loading model weights."""

import hashlib
import io

import onnx


def scan(stream, *, small_tensor_bytes=4096, metadata_field_bytes=1 << 20):
    """Scan a seekable ModelProto; seek over all large initializer payloads."""
    if not stream.seekable() or type(small_tensor_bytes) is not int or small_tensor_bytes < 0:
        raise ValueError("scan requires a seekable stream and nonnegative tensor budget")
    if type(metadata_field_bytes) is not int or not 1 <= metadata_field_bytes <= 1 << 20:
        raise ValueError("metadata field budget must be within 1 MiB")
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    read_bytes = 0

    def read(count):
        nonlocal read_bytes
        if not 0 <= count <= 1 << 20:
            raise ValueError("read budget exceeded")
        data = stream.read(count)
        read_bytes += len(data)
        if len(data) != count:
            raise ValueError("truncated protobuf field")
        return data

    def varint(end):
        value = 0
        for shift in range(0, 70, 7):
            if stream.tell() >= end:
                raise ValueError("truncated protobuf varint")
            byte = read(1)[0]
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("invalid protobuf varint")

    def fields(end):
        while stream.tell() < end:
            tag = varint(end)
            number, wire = tag >> 3, tag & 7
            if number == 0:
                raise ValueError("invalid protobuf field number")
            if wire == 0:
                yield number, wire, varint(end)
                continue
            if wire == 2:
                length = varint(end)
            elif wire in (1, 5):
                length = 8 if wire == 1 else 4
            else:
                raise ValueError("unsupported protobuf wire type")
            start = stream.tell()
            if start + length > end:
                raise ValueError("protobuf field exceeds parent boundary")
            yield number, wire, (start, length)
            stream.seek(start + length)

    def payload(location, budget):
        start, count = location
        if count > budget:
            raise ValueError("metadata or small tensor budget exceeded")
        stream.seek(start)
        return read(count)

    def tensor(location):
        start, count = location
        stream.seek(start)
        value = dict(name="", dims=[], data_type=0, raw_bytes=0)
        raw = None
        for number, wire, item in fields(start + count):
            if number == 1 and wire == 0:
                value["dims"].append(item)
            elif number == 1 and wire == 2:
                packed_start, packed_length = item
                stream.seek(packed_start)
                while stream.tell() < packed_start + packed_length:
                    value["dims"].append(varint(packed_start + packed_length))
            elif number == 2 and wire == 0:
                value["data_type"] = item
            elif number == 8 and wire == 2:
                value["name"] = payload(item, metadata_field_bytes).decode("utf-8")
            elif number == 9 and wire == 2:
                if raw is not None:
                    raise ValueError("duplicate raw tensor payload")
                raw = item
                value["raw_offset"], value["raw_bytes"] = item
        if not value["name"]:
            raise ValueError("initializer name is missing")
        if raw is not None and raw[1] <= small_tensor_bytes:
            value["sha256"] = hashlib.sha256(payload(raw, small_tensor_bytes)).hexdigest()
        return value

    result = dict(nodes=[], initializers=[], inputs=[], outputs=[], model_executed=False, weights_fully_read=False)
    graphs = 0
    for number, wire, item in fields(size):
        if number != 7 or wire != 2:
            continue
        graphs += 1
        if graphs != 1:
            raise ValueError("duplicate model graph")
        start, count = item
        stream.seek(start)
        for graph_field, graph_wire, location in fields(start + count):
            if graph_wire != 2:
                continue
            if graph_field == 1:
                node = onnx.NodeProto.FromString(payload(location, metadata_field_bytes))
                result["nodes"].append(dict(name=node.name, op_type=node.op_type, domain=node.domain,
                                            inputs=list(node.input), outputs=list(node.output)))
            elif graph_field == 5:
                result["initializers"].append(tensor(location))
            elif graph_field in (11, 12):
                info = onnx.ValueInfoProto.FromString(payload(location, metadata_field_bytes))
                result["inputs" if graph_field == 11 else "outputs"].append(info.name)
    if graphs != 1:
        raise ValueError("model graph is missing")
    result.update(model_bytes=size, read_bytes=read_bytes)
    return result

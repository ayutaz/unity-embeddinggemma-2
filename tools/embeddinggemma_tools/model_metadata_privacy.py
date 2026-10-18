"""Read-only model metadata checks; opaque numerical tensor bytes stay opaque."""
from pathlib import Path,PurePosixPath,PureWindowsPath
import zipfile

from .exported_program_distribution import _members,_json,_strings,_text_members,_privacy,METADATA_LIMIT


def _pt2(path):
    with zipfile.ZipFile(path) as archive:
        _,name=_members(archive)
        if not 0<archive.getinfo(name).file_size<=METADATA_LIMIT:
            raise ValueError('Unsupported model metadata size')
        _strings(_json(archive.read(name)))
        _text_members(archive,name)


def _onnx(path):
    import onnx
    model=onnx.load_model(str(path),load_external_data=False)
    def visit(message,depth=0):
        if depth>128:raise ValueError('Unsupported model metadata depth')
        if isinstance(message,onnx.TensorProto) and message.external_data:
            locations=[entry.value for entry in message.external_data if entry.key=='location']
            if len(locations)!=1:raise ValueError('Ambiguous external data location')
            name=locations[0];posix=PurePosixPath(name);windows=PureWindowsPath(name)
            if not name or posix.is_absolute() or windows.drive or windows.root or '..' in posix.parts or '..' in windows.parts:
                raise ValueError('External data must stay relative')
        if isinstance(message,onnx.AttributeProto):
            if message.type==onnx.AttributeProto.STRING:_privacy(message.s.decode('utf-8'))
            if message.type==onnx.AttributeProto.STRINGS:
                for item in message.strings:_privacy(item.decode('utf-8'))
        for field,value in message.ListFields():
            if field.type not in (field.TYPE_STRING,field.TYPE_MESSAGE):continue
            values=value if field.is_repeated else [value]
            for item in values:
                if field.type==field.TYPE_STRING:_privacy(item)
                else:visit(item,depth+1)
    visit(model)


def audit_model_metadata(path):
    """Return only fixed rule/path labels, never model strings or error details."""
    try:
        suffix=Path(path).suffix.lower()
        if suffix=='.pt2':_pt2(path)
        elif suffix=='.onnx':_onnx(path)
        else:raise ValueError('Unsupported model metadata format')
        return []
    except Exception:
        return [dict(path='<artifact-model>',line=1,rule='unreviewed-or-private-model-metadata')]

from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class InferInput(_message.Message):
    __slots__ = ["prompts"]
    PROMPTS_FIELD_NUMBER: _ClassVar[int]
    prompts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, prompts: _Optional[_Iterable[str]] = ...) -> None: ...

class InferOutput(_message.Message):
    __slots__ = ["rank", "texts"]
    RANK_FIELD_NUMBER: _ClassVar[int]
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    rank: int
    texts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, rank: _Optional[int] = ..., texts: _Optional[_Iterable[str]] = ...) -> None: ...

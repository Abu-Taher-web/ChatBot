# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: inference.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'inference.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\"V\n\x10IntermediateData\x12\x11\n\tinput_ids\x18\x01 \x01(\x0c\x12\x16\n\x0e\x61ttention_mask\x18\x02 \x01(\x0c\x12\x17\n\x0fpast_key_values\x18\x03 \x01(\x0c\"(\n\x10GenerationResult\x12\x14\n\x0c\x63ontinuation\x18\x01 \x01(\t2N\n\x10InferenceService\x12:\n\x12\x43ontinueGeneration\x12\x11.IntermediateData\x1a\x11.GenerationResultb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_INTERMEDIATEDATA']._serialized_start=19
  _globals['_INTERMEDIATEDATA']._serialized_end=105
  _globals['_GENERATIONRESULT']._serialized_start=107
  _globals['_GENERATIONRESULT']._serialized_end=147
  _globals['_INFERENCESERVICE']._serialized_start=149
  _globals['_INFERENCESERVICE']._serialized_end=227
# @@protoc_insertion_point(module_scope)

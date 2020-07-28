LIBRARY()

PROVIDES(protobuf)

LICENSE(
    BSD3
)



NO_COMPILER_WARNINGS()

SUPPRESSIONS(tsan.supp)

ADDINCL(
    GLOBAL contrib/libs/protobuf
    GLOBAL contrib/libs/protobuf/src
    GLOBAL FOR proto contrib/libs/protobuf/src
)

JOINSRC()

IF (OS_ANDROID)
    EXTRALIBS(-llog)
ENDIF()

CFLAGS(-DHAVE_ZLIB)

PEERDIR(contrib/libs/zlib)

SRCS(
    src/google/protobuf/any.cc
    src/google/protobuf/any.pb.cc
    src/google/protobuf/api.pb.cc
    src/google/protobuf/arena.cc
    src/google/protobuf/arenastring.cc
    src/google/protobuf/descriptor.cc
    src/google/protobuf/descriptor.pb.cc
    src/google/protobuf/descriptor_database.cc
    src/google/protobuf/duration.pb.cc
    src/google/protobuf/dynamic_message.cc
    src/google/protobuf/empty.pb.cc
    src/google/protobuf/extension_set.cc
    src/google/protobuf/extension_set_heavy.cc
    src/google/protobuf/field_mask.pb.cc
    src/google/protobuf/generated_message_reflection.cc
    src/google/protobuf/generated_message_table_driven.cc
    src/google/protobuf/generated_message_table_driven_lite.cc
    src/google/protobuf/generated_message_util.cc
    src/google/protobuf/io/coded_stream.cc
    src/google/protobuf/io/gzip_stream.cc
    src/google/protobuf/io/printer.cc
    src/google/protobuf/io/strtod.cc
    src/google/protobuf/io/tokenizer.cc
    src/google/protobuf/io/zero_copy_stream.cc
    src/google/protobuf/io/zero_copy_stream_impl.cc
    src/google/protobuf/io/zero_copy_stream_impl_lite.cc
    src/google/protobuf/json_util.cc
    src/google/protobuf/map_field.cc
    src/google/protobuf/message.cc
    src/google/protobuf/message_lite.cc
    src/google/protobuf/messagext.cc
    src/google/protobuf/messagext_lite.cc
    src/google/protobuf/reflection_ops.cc
    src/google/protobuf/repeated_field.cc
    src/google/protobuf/service.cc
    src/google/protobuf/source_context.pb.cc
    src/google/protobuf/struct.pb.cc
    src/google/protobuf/stubs/atomicops_internals_x86_gcc.cc
    src/google/protobuf/stubs/atomicops_internals_x86_msvc.cc
    src/google/protobuf/stubs/bytestream.cc
    src/google/protobuf/stubs/common.cc
    src/google/protobuf/stubs/int128.cc
    src/google/protobuf/stubs/io_win32.cc
    src/google/protobuf/stubs/mathlimits.cc
    src/google/protobuf/stubs/once.cc
    src/google/protobuf/stubs/status.cc
    src/google/protobuf/stubs/statusor.cc
    src/google/protobuf/stubs/stringpiece.cc
    src/google/protobuf/stubs/stringprintf.cc
    src/google/protobuf/stubs/structurally_valid.cc
    src/google/protobuf/stubs/strutil.cc
    src/google/protobuf/stubs/substitute.cc
    src/google/protobuf/stubs/time.cc
    src/google/protobuf/text_format.cc
    src/google/protobuf/timestamp.pb.cc
    src/google/protobuf/type.pb.cc
    src/google/protobuf/unknown_field_set.cc
    src/google/protobuf/util/delimited_message_util.cc
    src/google/protobuf/util/field_comparator.cc
    src/google/protobuf/util/field_mask_util.cc
    src/google/protobuf/util/internal/datapiece.cc
    src/google/protobuf/util/internal/default_value_objectwriter.cc
    src/google/protobuf/util/internal/error_listener.cc
    src/google/protobuf/util/internal/field_mask_utility.cc
    src/google/protobuf/util/internal/json_escaping.cc
    src/google/protobuf/util/internal/json_objectwriter.cc
    src/google/protobuf/util/internal/json_stream_parser.cc
    src/google/protobuf/util/internal/object_writer.cc
    src/google/protobuf/util/internal/proto_writer.cc
    src/google/protobuf/util/internal/protostream_objectsource.cc
    src/google/protobuf/util/internal/protostream_objectwriter.cc
    src/google/protobuf/util/internal/type_info.cc
    src/google/protobuf/util/internal/utility.cc
    src/google/protobuf/util/json_util.cc
    src/google/protobuf/util/message_differencer.cc
    src/google/protobuf/util/time_util.cc
    src/google/protobuf/util/type_resolver_util.cc
    src/google/protobuf/wire_format.cc
    src/google/protobuf/wire_format_lite.cc
    src/google/protobuf/wrappers.pb.cc
)

FILES(
    src/google/protobuf/any.proto
    src/google/protobuf/api.proto
    src/google/protobuf/descriptor.proto
    src/google/protobuf/duration.proto
    src/google/protobuf/empty.proto
    src/google/protobuf/field_mask.proto
    src/google/protobuf/source_context.proto
    src/google/protobuf/struct.proto
    src/google/protobuf/timestamp.proto
    src/google/protobuf/type.proto
    src/google/protobuf/wrappers.proto
)

END()

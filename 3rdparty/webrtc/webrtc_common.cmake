# Common configs for building WebRTC from source. Used in both native build
# and building inside docker.
#
# Exports:
# - get_webrtc_args(WEBRTC_ARGS) function
# - NINJA_TARGETS
# - EXTRA_WEBRTC_OBJS  # You have to define WEBRTC_NINJA_ROOT before including this file

function(get_webrtc_args WEBRTC_ARGS)
    set(WEBRTC_ARGS "")

    # ABI selection
    if(GLIBCXX_USE_CXX11_ABI)
        set(WEBRTC_ARGS rtc_use_cxx11_abi=true\n${WEBRTC_ARGS})
    else()
        set(WEBRTC_ARGS rtc_use_cxx11_abi=false\n${WEBRTC_ARGS})
    endif()

    set(WEBRTC_ARGS rtc_include_tests=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_enable_protobuf=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_examples=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_tools=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS treat_warnings_as_errors=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_enable_libevent=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_libevent=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS use_sysroot=false\n${WEBRTC_ARGS})

    # Disable screen capturing
    set(WEBRTC_ARGS rtc_use_x11=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_use_pipewire=false\n${WEBRTC_ARGS})

    # Don't use libc++ (Clang), use libstdc++ (GNU)
    # https://stackoverflow.com/a/47384787/1255535
    set(WEBRTC_ARGS use_custom_libcxx=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS use_custom_libcxx_for_host=false\n${WEBRTC_ARGS})

    # Debug/Release
    if(WEBRTC_IS_DEBUG)
        set(WEBRTC_ARGS is_debug=true\n${WEBRTC_ARGS})
    else()
        set(WEBRTC_ARGS is_debug=false\n${WEBRTC_ARGS})
    endif()

    # H264 support
    set(WEBRTC_ARGS is_chrome_branded=true\n${WEBRTC_ARGS})

    # Sound support
    set(WEBRTC_ARGS rtc_include_pulse_audio=false\n${WEBRTC_ARGS})

    # Use clang for compilation
    set(WEBRTC_ARGS is_clang=false\n${WEBRTC_ARGS})

    # Use ccache if available, not recommended inside Docker
    find_program(CCACHE_BIN "ccache")
    if(CCACHE_BIN)
        set(WEBRTC_ARGS cc_wrapper="ccache"\n${WEBRTC_ARGS})
    endif()
  set(WEBRTC_ARGS ${WEBRTC_ARGS} PARENT_SCOPE)
endfunction()

# webrtc        -> libwebrtc.a
# other targets -> libwebrtc_extra.a
set(NINJA_TARGETS
    webrtc
    rtc_json
    jsoncpp
    builtin_video_decoder_factory
    builtin_video_encoder_factory
    peerconnection
    p2p_server_utils
    task_queue
    default_task_queue_factory
)

# Byproducts for ninja build, later packaged by CMake into libwebrtc_extra.a
if(NOT WEBRTC_NINJA_ROOT)
    message(FATAL_ERROR "Please define WEBRTC_NINJA_ROOT before including webrtc_common.cmake")
endif()
set(EXTRA_WEBRTC_OBJS
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_reader.o
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_value.o
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_writer.o
    ${WEBRTC_NINJA_ROOT}/obj/p2p/p2p_server_utils/stun_server.o
    ${WEBRTC_NINJA_ROOT}/obj/p2p/p2p_server_utils/turn_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_json/json.o


    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/crc32.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/data_rate_limiter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/async_invoker.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/async_tcp_socket.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/async_packet_socket.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/task_utils/pending_task_safety_flag/pending_task_safety_flag.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/task_utils/repeating_task/repeating_task.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/memory/fifo_buffer/fifo_buffer.o

    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/time/timestamp_extrapolator/timestamp_extrapolator.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/system/file_wrapper/file_wrapper.o
    ${WEBRTC_NINJA_ROOT}/obj/api/transport_api/transport.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtp_headers/rtp_headers.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtp_packet_info/rtp_packet_info.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtc_event_log_output_file/rtc_event_log_output_file.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtc_error/rtc_error.o
    ${WEBRTC_NINJA_ROOT}/obj/api/ice_transport_factory/ice_transport_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/api/media_stream_interface/media_stream_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/create_peerconnection_factory/create_peerconnection_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtp_parameters/rtp_parameters.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtp_parameters/media_types.o
    ${WEBRTC_NINJA_ROOT}/obj/api/neteq/tick_timer/tick_timer.o
    ${WEBRTC_NINJA_ROOT}/obj/api/neteq/default_neteq_controller_factory/default_neteq_controller_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/api/neteq/neteq_api/neteq.o
    ${WEBRTC_NINJA_ROOT}/obj/api/adaptation/resource_adaptation_api/resource.o
    ${WEBRTC_NINJA_ROOT}/obj/api/numerics/numerics/samples_stats_counter.o
    ${WEBRTC_NINJA_ROOT}/obj/api/units/data_rate/data_rate.o
    ${WEBRTC_NINJA_ROOT}/obj/api/units/frequency/frequency.o
    ${WEBRTC_NINJA_ROOT}/obj/api/units/timestamp/timestamp.o
    ${WEBRTC_NINJA_ROOT}/obj/api/units/time_delta/time_delta.o
    ${WEBRTC_NINJA_ROOT}/obj/api/units/data_size/data_size.o
    ${WEBRTC_NINJA_ROOT}/obj/api/crypto/options/crypto_options.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtc_event_log/rtc_event_log_factory/rtc_event_log_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtc_event_log/rtc_event_log/rtc_event_log.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtc_event_log/rtc_event_log/rtc_event.o
    ${WEBRTC_NINJA_ROOT}/obj/api/task_queue/task_queue/task_queue_base.o
    ${WEBRTC_NINJA_ROOT}/obj/api/task_queue/default_task_queue_factory/default_task_queue_factory_stdlib.o
    ${WEBRTC_NINJA_ROOT}/obj/api/transport/stun_types/stun.o
    ${WEBRTC_NINJA_ROOT}/obj/api/transport/network_control/network_types.o
    ${WEBRTC_NINJA_ROOT}/obj/api/transport/goog_cc/goog_cc_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/api/transport/field_trial_based_config/field_trial_based_config.o
    ${WEBRTC_NINJA_ROOT}/obj/api/transport/bitrate_settings/bitrate_settings.o
    ${WEBRTC_NINJA_ROOT}/obj/api/transport/rtp/dependency_descriptor/dependency_descriptor.o
)

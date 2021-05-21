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


    ${WEBRTC_NINJA_ROOT}/obj/stats/rtc_stats/rtc_stats_report.o
    ${WEBRTC_NINJA_ROOT}/obj/stats/rtc_stats/rtc_stats.o
    ${WEBRTC_NINJA_ROOT}/obj/stats/rtc_stats/rtcstats_objects.o
    ${WEBRTC_NINJA_ROOT}/obj/system_wrappers/metrics/metrics.o
    ${WEBRTC_NINJA_ROOT}/obj/system_wrappers/field_trial/field_trial.o
    ${WEBRTC_NINJA_ROOT}/obj/system_wrappers/system_wrappers/cpu_info.o
    ${WEBRTC_NINJA_ROOT}/obj/system_wrappers/system_wrappers/rtp_to_ntp_estimator.o
    ${WEBRTC_NINJA_ROOT}/obj/system_wrappers/system_wrappers/clock.o
    ${WEBRTC_NINJA_ROOT}/obj/system_wrappers/system_wrappers/cpu_features_linux.o
    ${WEBRTC_NINJA_ROOT}/obj/system_wrappers/system_wrappers/sleep.o
    ${WEBRTC_NINJA_ROOT}/obj/system_wrappers/system_wrappers/cpu_features.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_video/rtc_event_video_receive_stream_config.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_video/rtc_event_video_send_stream_config.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_stream_config/rtc_stream_config.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_pacing/rtc_event_alr_state.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_rtp_rtcp/rtc_event_rtp_packet_outgoing.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_rtp_rtcp/rtc_event_rtp_packet_incoming.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_rtp_rtcp/rtc_event_rtcp_packet_incoming.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_rtp_rtcp/rtc_event_rtcp_packet_outgoing.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/ice_log/rtc_event_ice_candidate_pair.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/ice_log/ice_logger.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/ice_log/rtc_event_dtls_transport_state.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/ice_log/rtc_event_ice_candidate_pair_config.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/ice_log/rtc_event_dtls_writable_state.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_log_impl_encoder/var_int.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_log_impl_encoder/rtc_event_log_encoder_common.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_log_impl_encoder/blob_encoding.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_log_impl_encoder/delta_encoding.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_bwe/rtc_event_probe_result_success.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_bwe/rtc_event_route_change.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_bwe/rtc_event_probe_cluster_created.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_bwe/rtc_event_bwe_update_loss_based.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_bwe/rtc_event_probe_result_failure.o
    ${WEBRTC_NINJA_ROOT}/obj/logging/rtc_event_bwe/rtc_event_bwe_update_delay_based.o
    ${WEBRTC_NINJA_ROOT}/obj/call/simulated_network/simulated_network.o
    ${WEBRTC_NINJA_ROOT}/obj/call/version/version.o
    ${WEBRTC_NINJA_ROOT}/obj/call/rtp_interfaces/rtp_config.o
    ${WEBRTC_NINJA_ROOT}/obj/call/call_interfaces/flexfec_receive_stream.o
    ${WEBRTC_NINJA_ROOT}/obj/call/call_interfaces/call_config.o
    ${WEBRTC_NINJA_ROOT}/obj/call/call_interfaces/syncable.o
    ${WEBRTC_NINJA_ROOT}/obj/call/fake_network/fake_network_pipe.o
    ${WEBRTC_NINJA_ROOT}/obj/call/rtp_receiver/rtx_receive_stream.o
    ${WEBRTC_NINJA_ROOT}/obj/call/rtp_receiver/rtp_demuxer.o
    ${WEBRTC_NINJA_ROOT}/obj/call/rtp_receiver/rtp_stream_receiver_controller.o
    ${WEBRTC_NINJA_ROOT}/obj/call/rtp_sender/rtp_transport_controller_send.o
    ${WEBRTC_NINJA_ROOT}/obj/call/rtp_sender/rtp_payload_params.o
    ${WEBRTC_NINJA_ROOT}/obj/call/rtp_sender/rtp_video_sender.o
    ${WEBRTC_NINJA_ROOT}/obj/call/call/call.o
    ${WEBRTC_NINJA_ROOT}/obj/call/call/call_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/call/call/degraded_call.o
    ${WEBRTC_NINJA_ROOT}/obj/call/call/receive_time_calculator.o
    ${WEBRTC_NINJA_ROOT}/obj/call/call/flexfec_receive_stream_impl.o
    ${WEBRTC_NINJA_ROOT}/obj/call/bitrate_configurator/rtp_bitrate_configurator.o
    ${WEBRTC_NINJA_ROOT}/obj/call/bitrate_allocator/bitrate_allocator.o
    ${WEBRTC_NINJA_ROOT}/obj/call/video_stream_api/video_receive_stream.o
    ${WEBRTC_NINJA_ROOT}/obj/call/video_stream_api/video_send_stream.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/resource_adaptation_processor.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/video_stream_adapter.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/degradation_preference_provider.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/video_stream_input_state_provider.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/video_stream_input_state.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/video_source_restrictions.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/resource_adaptation_processor_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/encoder_settings.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/broadcast_resource_listener.o
    ${WEBRTC_NINJA_ROOT}/obj/call/adaptation/resource_adaptation/adaptation_constraint.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/video_rtp_track_source/video_rtp_track_source.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtp_transceiver/rtp_transceiver.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtp_sender/rtp_sender.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/video_track/video_track.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtp_parameters_conversion/rtp_parameters_conversion.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtp_transmission_manager/rtp_transmission_manager.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/video_track_source/video_track_source.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/video_rtp_receiver/video_rtp_receiver.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peer_connection_message_handler/peer_connection_message_handler.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtp_receiver/rtp_receiver.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/usage_pattern/usage_pattern.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/media_stream/media_stream.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/jitter_buffer_delay/jitter_buffer_delay.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/media_protocol_names/media_protocol_names.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/transceiver_list/transceiver_list.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/connection_context/connection_context.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/dtmf_sender/dtmf_sender.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/channel_manager.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/jsep_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/dtls_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/dtls_srtp_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/rtp_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/sctp_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/sctp_data_channel_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/srtp_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/session_description.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/rtp_media_utils.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/transport_stats.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/sctp_utils.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/channel.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/srtp_session.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/srtp_filter.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/ice_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/simulcast_description.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/external_hmac.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/jsep_transport_controller.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/media_session.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/rtc_pc_base/rtcp_mux_filter.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/sdp_serializer.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/sdp_utils.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/data_channel_controller.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/peer_connection.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/media_stream_observer.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/jsep_ice_candidate.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/webrtc_sdp.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/webrtc_session_description_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/jsep_session_description.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/ice_server_parsing.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/peer_connection_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/sdp_offer_answer.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/sctp_data_channel.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/rtp_data_channel.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/rtc_stats_collector.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/rtc_stats_traversal.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/data_channel_utils.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/track_media_info_map.o
    ${WEBRTC_NINJA_ROOT}/obj/pc/peerconnection/stats_collector.o
    ${WEBRTC_NINJA_ROOT}/obj/media/rtc_simulcast_encoder_adapter/simulcast_encoder_adapter.o
    ${WEBRTC_NINJA_ROOT}/obj/media/rtc_vp9_profile/vp9_profile.o
    ${WEBRTC_NINJA_ROOT}/obj/media/rtc_data/sctp_transport.o
    ${WEBRTC_NINJA_ROOT}/obj/media/rtc_encoder_simulcast_proxy/encoder_simulcast_proxy.o
    ${WEBRTC_NINJA_ROOT}/obj/media/rtc_h264_profile_id/h264_profile_level_id.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/weak_ptr/weak_ptr.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/stringutils/string_format.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/stringutils/string_builder.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/stringutils/string_utils.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/stringutils/string_to_number.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/stringutils/string_encode.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/threading/default_socket_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/threading/network_monitor.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/threading/message_handler.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/threading/async_resolver.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/threading/thread.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/threading/network_monitor_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/threading/physical_socket_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/socket_address/socket_address.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/timeutils/system_time.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/timeutils/time_utils.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_task_queue_stdlib/task_queue_stdlib.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_task_queue/task_queue.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/socket/socket.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_numerics/event_based_exponential_moving_average.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_numerics/moving_average.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_numerics/exp_filter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/copy_on_write_buffer.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/byte_buffer.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/event_tracer.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/buffer_queue.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/rate_tracker.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/rate_statistics.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/timestamp_aligner.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/bit_buffer.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/zero_memory.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/random.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/race_checker.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/sample_counter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/histogram_percentile_counter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_approved/location.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/nat_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/nat_socket_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/nat_types.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/virtual_socket_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/server_socket_adapters.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/test_echo_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/test_utils.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/fake_clock.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/socket_stream.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/proxy_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/fake_ssl_identity.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/cpu_time.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/memory_usage.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/memory_stream.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base_tests_utils/firewall_socket_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_event/event.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_json/json.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_operations_chain/operations_chain.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/platform_thread_types/platform_thread_types.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/platform_thread/platform_thread.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/null_socket_server/null_socket_server.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/ip_address/ip_address.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/logging/logging.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/net_helpers/net_helpers.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/network_constants/network_constants.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/checks/checks.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/criticalsection/recursive_critical_section.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/callback_list/callback_list.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/async_resolver_interface/async_resolver_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/async_socket/async_socket.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/boringssl_certificate.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/boringssl_identity.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/log_sinks.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/ifaddrs_converter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/unique_id_generator.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/stream.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/ssl_stream_adapter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/ssl_identity.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/proxy_info.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/rtc_certificate_generator.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/ssl_fingerprint.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/openssl_stream_adapter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/socket_adapters.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/ssl_certificate.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/ssl_adapter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/openssl_session_cache.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/rtc_certificate.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/openssl_utility.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/openssl_adapter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/socket_address_pair.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/openssl_key_pair.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/openssl_digest.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/http_common.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/network.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/network_route.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/helpers.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/message_digest.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/net_helper.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/file_rotating_stream.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/async_udp_socket.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/crypt_string.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/crc32.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/data_rate_limiter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/async_invoker.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/async_tcp_socket.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_base/async_packet_socket.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rate_limiter/rate_limiter.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/stable_target_rate_experiment/stable_target_rate_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/rate_control_settings/rate_control_settings.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/quality_scaler_settings/quality_scaler_settings.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/rtt_mult_experiment/rtt_mult_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/quality_scaling_experiment/quality_scaling_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/min_video_bitrate_experiment/min_video_bitrate_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/normalize_simulcast_size_experiment/normalize_simulcast_size_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/quality_rampup_experiment/quality_rampup_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/keyframe_interval_settings_experiment/keyframe_interval_settings.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/alr_experiment/alr_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/cpu_speed_experiment/cpu_speed_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/balanced_degradation_settings/balanced_degradation_settings.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/jitter_upper_bound_experiment/jitter_upper_bound_experiment.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/encoder_info_settings/encoder_info_settings.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/field_trial_parser/field_trial_list.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/field_trial_parser/field_trial_units.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/field_trial_parser/field_trial_parser.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/experiments/field_trial_parser/struct_parameters_parser.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/task_utils/pending_task_safety_flag/pending_task_safety_flag.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/task_utils/repeating_task/repeating_task.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/memory/fifo_buffer/fifo_buffer.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/memory/aligned_malloc/aligned_malloc.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/synchronization/sequence_checker_internal/sequence_checker_internal.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/synchronization/mutex/mutex.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/synchronization/yield_policy/yield_policy.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/synchronization/yield/yield.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/network/sent_packet/sent_packet.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/third_party/sigslot/sigslot/sigslot.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/third_party/base64/base64/base64.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/time/timestamp_extrapolator/timestamp_extrapolator.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/system/file_wrapper/file_wrapper.o
    ${WEBRTC_NINJA_ROOT}/obj/api/transport_api/transport.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtp_headers/rtp_headers.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtp_packet_info/rtp_packet_info.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtc_event_log_output_file/rtc_event_log_output_file.o
    ${WEBRTC_NINJA_ROOT}/obj/api/rtc_error/rtc_error.o
    ${WEBRTC_NINJA_ROOT}/obj/api/ice_transport_factory/ice_transport_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/api/media_stream_interface/media_stream_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/data_channel_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/rtp_receiver_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/dtls_transport_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/stats_types.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/proxy.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/rtp_transceiver_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/rtp_sender_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/sctp_transport_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/peer_connection_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/candidate.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/jsep_ice_candidate.o
    ${WEBRTC_NINJA_ROOT}/obj/api/libjingle_peerconnection_api/jsep.o
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
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_frame_i010/i010_buffer.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_bitrate_allocator/video_bitrate_allocator.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_bitrate_allocation/video_bitrate_allocation.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/encoded_frame/encoded_frame.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_adaptation/video_adaptation_counters.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_frame_metadata/video_frame_metadata.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_rtp_headers/color_space.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_rtp_headers/video_content_type.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_rtp_headers/video_timing.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_rtp_headers/hdr_metadata.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/builtin_video_bitrate_allocator_factory/builtin_video_bitrate_allocator_factory.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_frame/video_frame.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_frame/video_frame_buffer.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_frame/nv12_buffer.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_frame/video_source_interface.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/video_frame/i420_buffer.o
    ${WEBRTC_NINJA_ROOT}/obj/api/video/encoded_image/encoded_image.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_video_header/rtp_video_header.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/flexfec_receiver.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/remote_ntp_time_estimator.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/video_rtp_depacketizer_vp9.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/video_rtp_depacketizer_generic.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/video_rtp_depacketizer_h264.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/fec_private_tables_random.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/ulpfec_header_reader_writer.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/video_rtp_depacketizer_vp8.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/video_rtp_depacketizer_raw.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/video_rtp_depacketizer_av1.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/ulpfec_receiver_impl.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/video_rtp_depacketizer.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/create_video_rtp_depacketizer.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/absolute_capture_time_sender.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/flexfec_sender.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/packet_sequencer.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/active_decode_targets_helper.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/absolute_capture_time_receiver.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/forward_error_correction.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/rtp_format_vp8.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/flexfec_header_reader_writer.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/dtmf_queue.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/rtp_format_vp9.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/rtp_header_extension_size.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/receive_statistics_impl.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/forward_error_correction_internal.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/source_tracker.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/rtp_sender_video.o
    ${WEBRTC_NINJA_ROOT}/obj/modules/rtp_rtcp/rtp_rtcp/fec_private_tables_bursty.o
)

/*
 *  Copyright 2012 The WebRTC Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <p2p/base/basic_packet_socket_factory.h>
#include <p2p/base/port_interface.h>
#include <p2p/base/turn_server.h>
#include <rtc_base/async_udp_socket.h>
#include <rtc_base/ip_address.h>
#include <rtc_base/socket_address.h>
#include <rtc_base/socket_server.h>
#include <rtc_base/thread.h>

#include <fstream>
#include <iostream>
#include <istream>
#include <map>
#include <string>
#include <utility>

namespace webrtc_examples {

std::map<std::string, std::string> ReadAuthFile(std::istream* s) {
    std::map<std::string, std::string> name_to_key;
    for (std::string line; std::getline(*s, line);) {
        const size_t sep = line.find('=');
        if (sep == std::string::npos) continue;
        char buf[32];
        size_t len = rtc::hex_decode(buf, sizeof(buf), line.data() + sep + 1,
                                     line.size() - sep - 1);
        if (len > 0) {
            name_to_key.emplace(line.substr(0, sep), std::string(buf, len));
        }
    }
    return name_to_key;
}

}  // namespace webrtc_examples

namespace {
const char kSoftware[] = "libjingle TurnServer";

class TurnFileAuth : public cricket::TurnAuthInterface {
public:
    explicit TurnFileAuth(std::map<std::string, std::string> name_to_key)
        : name_to_key_(std::move(name_to_key)) {}

    virtual bool GetKey(const std::string& username,
                        const std::string& realm,
                        std::string* key) {
        // File is stored as lines of <username>=<HA1>.
        // Generate HA1 via "echo -n "<username>:<realm>:<password>" | md5sum"
        auto it = name_to_key_.find(username);
        if (it == name_to_key_.end()) return false;
        *key = it->second;
        return true;
    }

private:
    const std::map<std::string, std::string> name_to_key_;
};

class AlwaysTrueAuth : public cricket::TurnAuthInterface {
public:
    explicit AlwaysTrueAuth() {}

    virtual bool GetKey(const std::string& username,
                        const std::string& realm,
                        std::string* key) {
        *key = "password";
        return true;
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    rtc::LogMessage::LogToDebug((rtc::LoggingSeverity)rtc::LS_VERBOSE);
    if (argc != 4) {
        std::cerr << "usage: turnserver int-addr ext-ip realm" << std::endl;
        return 1;
    }

    rtc::SocketAddress int_addr;
    if (!int_addr.FromString(argv[1])) {
        std::cerr << "Unable to parse IP address: " << argv[1] << std::endl;
        return 1;
    }

    rtc::IPAddress ext_addr;
    if (!IPFromString(argv[2], &ext_addr)) {
        std::cerr << "Unable to parse IP address: " << argv[2] << std::endl;
        return 1;
    }

    rtc::Thread* thread = rtc::Thread::Current();

    //////////////// Google Example
    // rtc::AsyncUDPSocket* int_socket =
    //         rtc::AsyncUDPSocket::Create(thread->socketserver(), int_addr);
    // if (!int_socket) {
    //     std::cerr << "Failed to create a UDP socket bound at"
    //               << int_addr.ToString() << std::endl;
    //     return 1;
    // }

    // cricket::TurnServer server(thread);
    // // std::fstream auth_file(argv[4], std::fstream::in);
    // // TurnFileAuth auth(auth_file.is_open()
    // //                           ? webrtc_examples::ReadAuthFile(&auth_file)
    // //                           : std::map<std::string, std::string>());
    // AlwaysTrueAuth auth;
    // server.set_realm(argv[3]);
    // server.set_software(kSoftware);
    // server.set_auth_hook(&auth);
    // server.AddInternalSocket(int_socket, cricket::PROTO_UDP);
    // server.SetExternalSocketFactory(new rtc::BasicPacketSocketFactory(),
    //                                 rtc::SocketAddress(ext_addr, 0));
    /////////////////

    ///////// webrtc-streamer example
    // Example usage:
    //
    // make DrawWebRTC -j && (cd bin/examples && \
    //   WEBRTC_STUN_SERVER="turn:user:password@$(curl -s ifconfig.me):3478" \
    //   WEBRTC_PUBLIC_IP=$(curl -s ifconfig.me) WEBRTC_IP=192.168.86.121 \
    //   ./DrawWebRTC)
    //
    // Internal address.
    std::unique_ptr<cricket::TurnServer> turn_server;
    turn_server.reset(new cricket::TurnServer(thread));
    // UDP.
    rtc::AsyncUDPSocket* udp_socket =
            rtc::AsyncUDPSocket::Create(thread->socketserver(), int_addr);
    if (udp_socket) {
        std::cout << "TURN Listening UDP at: " << int_addr.ToString()
                  << std::endl;
        turn_server->AddInternalSocket(udp_socket, cricket::PROTO_UDP);
    }
    // TCP.
    rtc::AsyncSocket* tcp_socket =
            thread->socketserver()->CreateAsyncSocket(AF_INET, SOCK_STREAM);
    if (tcp_socket) {
        std::cout << "TURN Listening TCP at: " << int_addr.ToString()
                  << std::endl;
        tcp_socket->Bind(int_addr);
        tcp_socket->Listen(5);
        turn_server->AddInternalServerSocket(tcp_socket, cricket::PROTO_TCP);
    }
    // External address.
    std::cout << "TURN external addr: " << ext_addr.ToString() << std::endl;
    turn_server->SetExternalSocketFactory(new rtc::BasicPacketSocketFactory(),
                                          rtc::SocketAddress(ext_addr, 0));
    /////////////////

    thread->Run();
    return 0;
}

/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** -------------------------------------------------------------------------*/

#pragma once

#include <api/video/i420_buffer.h>
#include <libyuv/convert.h>
#include <libyuv/video_common.h>
#include <media/base/video_broadcaster.h>
#include <media/base/video_common.h>
#include <modules/desktop_capture/desktop_capture_options.h>
#include <modules/desktop_capture/desktop_capturer.h>

#include <thread>

namespace open3d {
namespace visualization {
namespace webrtc_server {

class DesktopCapturer : public rtc::VideoSourceInterface<webrtc::VideoFrame>,
                        public webrtc::DesktopCapturer::Callback {
public:
    DesktopCapturer(const std::map<std::string, std::string>& opts)
        : width_(0), height_(0) {
        if (opts.find("width") != opts.end()) {
            width_ = std::stoi(opts.at("width"));
        }
        if (opts.find("height") != opts.end()) {
            height_ = std::stoi(opts.at("height"));
        }
    }
    bool Init() { return this->Start(); }
    virtual ~DesktopCapturer() { this->Stop(); }

    void CaptureThread();

    bool Start();
    void Stop();
    bool IsRunning() { return is_running_; }

    // overide webrtc::DesktopCapturer::Callback
    virtual void OnCaptureResult(webrtc::DesktopCapturer::Result result,
                                 std::unique_ptr<webrtc::DesktopFrame> frame);

    // overide rtc::VideoSourceInterface<webrtc::VideoFrame>
    virtual void AddOrUpdateSink(
            rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
            const rtc::VideoSinkWants& wants) {
        broadcaster_.AddOrUpdateSink(sink, wants);
    }

    virtual void RemoveSink(rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) {
        broadcaster_.RemoveSink(sink);
    }

protected:
    std::thread capture_thread_;
    std::unique_ptr<webrtc::DesktopCapturer> capturer_;
    int width_;
    int height_;
    bool is_running_;
    rtc::VideoBroadcaster broadcaster_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d

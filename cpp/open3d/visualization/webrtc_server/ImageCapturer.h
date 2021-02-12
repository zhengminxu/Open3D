/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** ScreenCapturer.h
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
#include <rtc_base/logging.h>

#include <thread>

#include "open3d/utility/Console.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class ImageCapturer : public rtc::VideoSourceInterface<webrtc::VideoFrame>,
                      public webrtc::DesktopCapturer::Callback {
public:
    ImageCapturer(const std::map<std::string, std::string>& opts)
        : m_width(0), m_height(0) {
        if (opts.find("width") != opts.end()) {
            m_width = std::stoi(opts.at("width"));
        }
        if (opts.find("height") != opts.end()) {
            m_height = std::stoi(opts.at("height"));
        }
    }
    bool Init() { return this->Start(); }
    virtual ~ImageCapturer() { this->Stop(); }

    void CaptureThread();

    bool Start();
    void Stop();
    bool IsRunning() { return m_isrunning; }

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
    std::thread m_capturethread;
    std::unique_ptr<webrtc::DesktopCapturer> m_capturer;
    int m_width;
    int m_height;
    bool m_isrunning;
    rtc::VideoBroadcaster broadcaster_;
};

class ImageWindowCapturer : public ImageCapturer {
public:
    ImageWindowCapturer(const std::string& url_,
                        const std::map<std::string, std::string>& opts)
        : ImageCapturer(opts) {
        utility::LogInfo("ImageWindowCapturer::url_: {}", url_);
        std::string url = "window://Open3D";
        const std::string windowprefix("window://");
        if (url.find(windowprefix) == 0) {
            m_capturer = webrtc::DesktopCapturer::CreateWindowCapturer(
                    webrtc::DesktopCaptureOptions::CreateDefault());

            if (m_capturer) {
                webrtc::DesktopCapturer::SourceList sourceList;
                if (m_capturer->GetSourceList(&sourceList)) {
                    const std::string windowtitle(
                            url.substr(windowprefix.length()));
                    for (auto source : sourceList) {
                        RTC_LOG(LS_ERROR)
                                << "ImageWindowCapturer source:" << source.id
                                << " title:" << source.title;
                        if (windowtitle == source.title) {
                            m_capturer->SelectSource(source.id);
                            break;
                        }
                    }
                }
            }
        }
    }
    static ImageWindowCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts) {
        std::unique_ptr<ImageWindowCapturer> capturer(
                new ImageWindowCapturer(url, opts));
        if (!capturer->Init()) {
            RTC_LOG(LS_WARNING) << "Failed to create ImageWindowCapturer";
            return nullptr;
        }
        return capturer.release();
    }
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d

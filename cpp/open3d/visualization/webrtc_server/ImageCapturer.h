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

#include "open3d/core/Tensor.h"
#include "open3d/t/io/ImageIO.h"
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
        t::geometry::Image im;
        t::io::ReadImage(
                "/home/yixing/repo/Open3D/cpp/open3d/visualization/"
                "webrtc_server/html/lena_color_640_480.jpg",
                im);
        im_buffer_ = core::Tensor::Zeros({im.GetRows(), im.GetCols(), 4},
                                         im.GetDtype());
        im_buffer_.Slice(2, 0, 1) = im.AsTensor().Slice(2, 2, 3);
        im_buffer_.Slice(2, 1, 2) = im.AsTensor().Slice(2, 1, 2);
        im_buffer_.Slice(2, 2, 3) = im.AsTensor().Slice(2, 0, 1);
    }
    bool Init() { return this->Start(); }
    virtual ~ImageCapturer() { this->Stop(); }

    void CaptureThread() {
        RTC_LOG(INFO) << "DesktopCapturer:Run start";
        while (IsRunning()) {
            m_capturer->CaptureFrame();
        }
        RTC_LOG(INFO) << "DesktopCapturer:Run exit";
    }

    bool Start() {
        m_isrunning = true;
        m_capturethread = std::thread(&ImageCapturer::CaptureThread, this);
        m_capturer->Start(this);
        return true;
    }
    void Stop() {
        m_isrunning = false;
        m_capturethread.join();
    }
    bool IsRunning() { return m_isrunning; }

    // overide webrtc::DesktopCapturer::Callback
    virtual void OnCaptureResult(webrtc::DesktopCapturer::Result result,
                                 std::unique_ptr<webrtc::DesktopFrame> frame) {
        RTC_LOG(INFO) << "ImageCapturer:OnCaptureResult";

        if (result == webrtc::DesktopCapturer::Result::SUCCESS) {
            // import numpy as np
            // import matplotlib.pyplot as plt
            // im = np.load("build/t_frame.npy")
            // im = np.flip(im[:, :, :3], axis=2)
            // print(im.shape)
            // print(im.dtype)
            // plt.imshow(im)
            int width = frame->stride() / webrtc::DesktopFrame::kBytesPerPixel;
            int height = frame->rect().height();
            // core::Tensor t_frame(static_cast<const uint8_t*>(frame->data()),
            //                      {height, width, 4}, core::Dtype::UInt8);
            // t_frame.Save("t_frame.npy");

            // width: 640,
            // height: 480
            // kBytesPerPixel: 4,
            // frame->stride(): 2560
            // utility::LogInfo(
            //         "width: {}, height: {}, kBytesPerPixel: {},
            //         frame->stride():
            //         "
            //         "{}",
            //         width, height, webrtc::DesktopFrame::kBytesPerPixel,
            //         frame->stride());

            rtc::scoped_refptr<webrtc::I420Buffer> I420buffer =
                    webrtc::I420Buffer::Create(width, height);

            // frame->data()
            const int conversionResult = libyuv::ConvertToI420(
                    static_cast<const uint8_t*>(im_buffer_.GetDataPtr()), 0,
                    I420buffer->MutableDataY(), I420buffer->StrideY(),
                    I420buffer->MutableDataU(), I420buffer->StrideU(),
                    I420buffer->MutableDataV(), I420buffer->StrideV(), 0, 0,
                    width, height, I420buffer->width(), I420buffer->height(),
                    libyuv::kRotate0, ::libyuv::FOURCC_ARGB);

            if (conversionResult >= 0) {
                webrtc::VideoFrame videoFrame(
                        I420buffer, webrtc::VideoRotation::kVideoRotation_0,
                        rtc::TimeMicros());
                if ((m_height == 0) && (m_width == 0)) {
                    broadcaster_.OnFrame(videoFrame);

                } else {
                    int height = m_height;
                    int width = m_width;
                    if (height == 0) {
                        height = (videoFrame.height() * width) /
                                 videoFrame.width();
                    } else if (width == 0) {
                        width = (videoFrame.width() * height) /
                                videoFrame.height();
                    }
                    int stride_y = width;
                    int stride_uv = (width + 1) / 2;
                    rtc::scoped_refptr<webrtc::I420Buffer> scaled_buffer =
                            webrtc::I420Buffer::Create(width, height, stride_y,
                                                       stride_uv, stride_uv);
                    scaled_buffer->ScaleFrom(
                            *videoFrame.video_frame_buffer()->ToI420());
                    webrtc::VideoFrame frame = webrtc::VideoFrame(
                            scaled_buffer, webrtc::kVideoRotation_0,
                            rtc::TimeMicros());

                    broadcaster_.OnFrame(frame);
                }
            } else {
                RTC_LOG(LS_ERROR)
                        << "DesktopCapturer:OnCaptureResult conversion error:"
                        << conversionResult;
            }

        } else {
            RTC_LOG(LS_ERROR)
                    << "DesktopCapturer:OnCaptureResult capture error:"
                    << (int)result;
        }
    }

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
    core::Tensor im_buffer_;  // Currently BGRA. TODO: make this RGB only.
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

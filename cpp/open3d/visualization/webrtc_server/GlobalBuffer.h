// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once
#include <mutex>

#include "open3d/core/Tensor.h"
#include "open3d/t/io/ImageIO.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class GlobalBuffer {
public:
    static GlobalBuffer& GetInstance() {
        static GlobalBuffer instance;
        return instance;
    }

    std::shared_ptr<core::Tensor> Read() {
        {
            // std::lock_guard<std::mutex> lock(frame_mutex_);
            return frame_;
        }
    }

    void Write(const std::shared_ptr<core::Tensor>& new_frame) {
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            new_frame->AssertShape(frame_->GetShape());
            new_frame->AssertDtype(frame_->GetDtype());
            new_frame->AssertDevice(frame_->GetDevice());
            frame_ = new_frame;
        }
    }

private:
    GlobalBuffer() {
        core::Tensor two_fifty_five =
                core::Tensor::Ones({}, core::Dtype::UInt8) * 255;
        frame_ = std::make_shared<core::Tensor>(core::SizeVector({480, 640, 3}),
                                                core::Dtype::UInt8);
        frame_->Fill(0);
        frame_->Slice(0, 0, 160, 1).Slice(2, 0, 1, 1) = two_fifty_five;
        frame_->Slice(0, 160, 320, 1).Slice(2, 1, 2, 1) = two_fifty_five;
        frame_->Slice(0, 320, 480, 1).Slice(2, 2, 3, 1) = two_fifty_five;
    }

    virtual ~GlobalBuffer() {}

    std::shared_ptr<core::Tensor> frame_ = nullptr;
    std::mutex frame_mutex_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d

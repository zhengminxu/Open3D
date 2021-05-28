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

#include <cstdlib>

#include "open3d/Open3D.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/rendering/Open3DScene.h"

// TODO: edit Open3D.h.in
#include "open3d/visualization/webrtc_server/WebRTCWindowSystem.h"

using namespace open3d;

// TODO: remove hard-coded path.
const std::string TEST_DIR = "../../../examples/test_data";

// Create and add a window to gui::Application, but do not run it yet.
void AddDrawWindow(
        const std::vector<std::shared_ptr<geometry::Geometry3D>> &geometries,
        const std::string &window_name = "Open3D",
        int width = 1024,
        int height = 768,
        const std::vector<visualization::DrawAction> &actions = {}) {
    std::vector<visualization::DrawObject> objects;
    objects.reserve(geometries.size());
    for (size_t i = 0; i < geometries.size(); ++i) {
        std::stringstream name;
        name << "Object " << (i + 1);
        objects.emplace_back(name.str(), geometries[i]);
    }

    auto &o3d_app = visualization::gui::Application::GetInstance();
    o3d_app.Initialize();

    auto draw = std::make_shared<visualization::visualizer::O3DVisualizer>(
            window_name, width, height);
    // draw->ShowMenu(true);
    // auto scene = draw->GetScene();
    // scene->SetLighting(open3d::visualization::rendering::Open3DScene::LightingProfile::NO_SHADOWS,
    // {-0.331, 0.839, 0.431});
    draw->SetHDRI("streetlamp");
    // draw->impl_->SetIBL("");

    auto hidefunc =
            [](open3d::visualization::visualizer::O3DVisualizer &o3dvis) {
                o3dvis.ShowMenu(false);
            };

    draw->AddAction("HideMenu", hidefunc);
    draw->AddAction("eMenu", hidefunc);
    draw->AddAction("eenu", hidefunc);
    draw->AddAction("eMeu", hidefunc);
    draw->AddAction("eMenu", hidefunc);
    draw->AddAction("eMnu", hidefunc);
    draw->AddAction("eenu", hidefunc);

    for (auto &o : objects) {
        if (o.geometry) {
            draw->AddGeometry(o.name, o.geometry);
        } else {
            draw->AddGeometry(o.name, o.tgeometry);
        }
        draw->ShowGeometry(o.name, o.is_visible);
    }
    for (auto &act : actions) {
        draw->AddAction(act.name, act.callback);
    }
    draw->ResetCameraToDefault();
    visualization::gui::Application::GetInstance().AddWindow(draw);

    draw.reset();  // so we don't hold onto the pointer after Run() cleans up
}

// Create a window with an empty box and a custom action button for adding a
// new visualization vindow.
void DrawPCD(const std::string &filename) {
    geometry::PointCloud pcd;
    io::ReadPointCloud(filename, pcd);
    pcd.EstimateNormals();
    auto pcd_ptr = std::make_shared<geometry::PointCloud>(pcd);

    AddDrawWindow({pcd_ptr}, filename, 600, 400);
}

int main(int argc, char **argv) {
    if (!utility::filesystem::DirectoryExists(TEST_DIR)) {
        utility::LogError(
                "This example needs to be run from the build directory, "
                "test_dir: {}",
                TEST_DIR);
    }

    visualization::webrtc_server::WebRTCWindowSystem::GetInstance()
            ->EnableWebRTC();

    // Uncomment this line to see more WebRTC loggings
    // utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    DrawPCD(TEST_DIR + "/open3d_downloads/redwood/bedroom.ply");
    DrawPCD(TEST_DIR + "/open3d_downloads/redwood/apartment.ply");
    DrawPCD(TEST_DIR + "/open3d_downloads/redwood/boardroom.ply");
    DrawPCD(TEST_DIR + "/open3d_downloads/redwood/loft.ply");

    visualization::gui::Application::GetInstance().Run();
}

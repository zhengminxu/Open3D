//-----------------------------------------------------------------------------
#include <Open3D/Open3D.h>
#include <Open3D/GUI/Application.h>
#include <Open3D/GUI/Native.h>
#include <Open3D/Visualization/Rendering/Camera.h>
#include <Open3D/Visualization/Rendering/Scene.h>
#include <Open3D/Visualization/Rendering/Filament/FilamentEngine.h>
#include <Open3D/Visualization/Rendering/Filament/FilamentResourceManager.h>
#include <Open3D/Visualization/Rendering/Filament/FilamentRenderer.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA 1
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <iostream>
#include <map>

using namespace open3d;
using namespace visualization;

static float MIN_FAR_PLANE = 100.0f;
static float NEAR_PLANE = 0.1f;
static float VERTICAL_FOV = 60.0f;

struct WindowData {
    int width;
    int height;
    FilamentRenderer *renderer;
    Scene *scene;
    View *view;
};

std::map<GLFWwindow*, WindowData> gWindowData;

void CreateFilamentScene(WindowData& data, std::shared_ptr<geometry::PointCloud> cloud) {
    // Create scene
    auto sceneId = data.renderer->CreateScene();
    auto scene = data.renderer->GetScene(sceneId);
    auto viewId = scene->AddView(0, 0, 1, 1);
    auto view = scene->GetView(viewId);
    view->SetClearColor({1, 1, 1});
    view->SetViewport(0, 0, data.width, data.height);

    data.scene = scene;
    data.view = view;

    // Create light
/*    visualization::LightDescription lightDescription;
    lightDescription.intensity = 100000;
    lightDescription.direction = {0.577f, -0.577f, -0.577f};
    lightDescription.castShadows = true;
    lightDescription.customAttributes["custom_type"] = "SUN";
    auto sun = scene->AddLight(lightDescription);
*/

    std::string rsrcPath = gui::Application::GetInstance().GetResourcePath();
    auto iblPath = rsrcPath + "/default_ibl.ktx";
    auto ibl =
            data.renderer->AddIndirectLight(ResourceLoadRequest(iblPath.data()));
    scene->SetIndirectLight(ibl);
    scene->SetIndirectLightIntensity(65000);
    //scene->SetIndirectLightRotation(lightingProfile.iblRotation);

    // Create materials
    auto litPath = rsrcPath + "/defaultLit.filamat";
    auto litMaterial = data.renderer->AddMaterial(
            visualization::ResourceLoadRequest(litPath.data()));

//    auto litInstance = data.renderer->AddMaterialInstance(litMaterial);
    auto litInstance = data.renderer->ModifyMaterial(litMaterial)
                    .SetColor("baseColor", Eigen::Vector4f{0.8f, 0.8f, 0.8f, 1.0f})
                    .SetParameter("roughness", 0.7f)
                    .SetParameter("metallic", 0.0f)
                    .SetParameter("reflectance", 0.5f)
                    .SetParameter("clearCoat", 0.2f)
                    .SetParameter("clearCoatRoughness", 0.2f)
                    .SetParameter("anisotropy", 0.0f)
                    .SetParameter("pointSize", 5)
                    .Finish();

    //auto unlitPath = rsrcPath + "/defaultUnlit.filamat";
    //impl_->hUnlitMaterial = GetRenderer().AddMaterial(
    //        visualization::ResourceLoadRequest(unlitPath.data()));

    geometry::AxisAlignedBoundingBox bounds;

    auto g3 = std::static_pointer_cast<const geometry::Geometry3D>(cloud);
    auto handle = scene->AddGeometry(*g3, litInstance);
    bounds += scene->GetEntityBoundingBox(handle);

    float aspect = 1.0f;
    if (data.height > 0) {
        aspect = float(data.width) / float(data.height);
    }
    auto far = std::max(MIN_FAR_PLANE, float(2.0 * bounds.GetExtent().norm()));
    view->GetCamera()->SetProjection(VERTICAL_FOV, aspect, NEAR_PLANE, far,
                                     Camera::FovType::Vertical);

    auto boundsMax = bounds.GetMaxBound();
    auto maxDim =
            std::max(boundsMax.x(), std::max(boundsMax.y(), boundsMax.z()));
    maxDim = 1.5f * maxDim;
    auto center = bounds.GetCenter().cast<float>();
    auto eye = Eigen::Vector3f(center.x(), center.y(), maxDim);
    auto up = Eigen::Vector3f(0, 1, 0);
    view->GetCamera()->LookAt(center, eye, up);
}

void OnResize(GLFWwindow *w, int osWidth, int osHeight) {
    float xscale, yscale;
    glfwGetWindowContentScale(w, &xscale, &yscale);
    WindowData &data = gWindowData[w];
    data.width = osWidth * xscale;
    data.height = osHeight * yscale;

    std::cout << "[debug] resize: " << osWidth << ", " << osHeight << " "
              << data.width << ", " << data.height << std::endl;

    data.renderer->UpdateSwapChain();

    data.view->SetViewport(0, 0, data.width, data.height);
}

void OnDraw(GLFWwindow *w) {
    WindowData &data = gWindowData[w];
    data.renderer->BeginFrame();
    data.renderer->Draw();
    data.renderer->EndFrame();
}

void OnKeyEvent(GLFWwindow *w, int key, int scancode, int action, int mods) {
    std::cout << "[key] " << key << ", " << scancode << ", " << action << std::endl;
    if (action == GLFW_RELEASE) {
        return;
    }

    auto move = [w](const Eigen::Vector3f& v) {
        WindowData &data = gWindowData[w];
        auto matrix = data.view->GetCamera()->GetModelMatrix();
        matrix.translate(v);
        data.view->GetCamera()->SetModelMatrix(matrix);
    };

    const float dist = 0.1;
    if (key == 'A') {
        move({-dist, 0, 0});
    } else if (key == 'D') {
        move({dist, 0, 0});
    } else if (key == 'W') {
        move({0, 0, -dist});
    } else if (key == 'S') {
        move({0, 0, dist});
    } else if (key == 'Q') {
        move({0, dist, 0});
    } else if (key == 'Z') {
        move({0, -dist, 0});
    }
    OnDraw(w);
}

bool CreateWindow(const char *path) {
    auto cloud = std::make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(path, *cloud)) {
        std::cerr << "[ERROR] could not read point cloud '" << path << std::endl;
        return false;
    }
    cloud->NormalizeNormals();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);

    int width = 1600;
    int height = 800;
    GLFWwindow* w = glfwCreateWindow(width, height, "Open3D - GLFWTest", NULL, NULL);
    if (!w) {
        std::cerr << "[ERROR] could not create window" << std::endl;
        return false;
    }

    gWindowData[w] = WindowData();
    WindowData &data = gWindowData[w];
    float xscale, yscale;
    glfwGetWindowContentScale(w, &xscale, &yscale);
    data.width = width * xscale;
    data.height = height * yscale;

    auto& engine = EngineInstance::GetInstance();
    auto& resourceManager = EngineInstance::GetResourceManager();
    data.renderer = new FilamentRenderer(engine, open3d::gui::GetNativeDrawable(w), resourceManager);
    data.renderer->UpdateSwapChain();
    CreateFilamentScene(data, cloud);

    glfwSetWindowRefreshCallback(w, OnDraw);
    glfwSetWindowSizeCallback(w, OnResize);
    glfwSetKeyCallback(w, OnKeyEvent);

    while (!glfwWindowShouldClose(w)) {
        glfwWaitEvents();
    }

    glfwDestroyWindow(w);
    return true;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << "pointcloud" << std::endl;
        return 0;
    }
    const char *path = argv[1];

    gui::Application::GetInstance().Initialize(argc, argv);

    if (!glfwInit()) {
        std::cerr << "[ERROR] could not initialize GLFW" << std::endl;
        return 1;
    }

    CreateWindow(path);

    glfwTerminate();
}

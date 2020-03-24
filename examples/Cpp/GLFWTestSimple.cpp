//-----------------------------------------------------------------------------
#include <Open3D/GUI/Native.h>

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h> // ??
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <filament/utils/EntityManager.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA 1
#include <GLFW/glfw3native.h>

#include <random.h>
#include <GL/gl.h>

#include <algorithm>
#include <iostream>
#include <map>

#define USE_FILAMENT 1

#if USE_FILAMENT
static float MIN_FAR_PLANE = 100.0f;
static float NEAR_PLANE = 0.1f;
static float VERTICAL_FOV = 60.0f;
#endif // USE_FILAMENT

struct WindowData {
    int width;
    int height;
#if USE_FILAMENT
    filament::Engine *engine = nullptr;
    filament::Renderer *renderer = nullptr;
    filament::SwapChain *swapChain = nullptr;
    filament::Scene *scene = nullptr;
    filament::View *view = nullptr;
#endif // USE_FILAMENT
};

std::map<GLFWwindow*, WindowData> gWindowData;

#if USE_FILAMENT
void CreateSwapChain(GLFWwindow *w) {
    WindowData &data = gWindowData[w];
    if (data.swapChain) {
        data.engine->destroy(data.swapChain);
        std::cout << "[debug] destroyed swap chain" << std::endl;
    }
    data.swapChain = data.engine->createSwapChain(open3d::gui::GetNativeDrawable(w));
    std::cout << "[debug] created swap chain" << std::endl;
}

void SetRandomBGColor(WindowData &data) {
    float red = double(random()) / double(RAND_MAX);
    float green = double(random()) / double(RAND_MAX);
    float blue = double(random()) / double(RAND_MAX);

    data.view->setClearColor({ red, green, blue, 1.0f });
}

void CreateFilamentScene(WindowData& data) {
    // Create scene
    auto scene = data.engine->createScene();
    auto view = data.engine->createView();
    view->setViewport({0, 0, uint32_t(data.width), uint32_t(data.height)});
    view->setClearColor({ 1, 1, 1, 1});
    view->setScene(scene);

    auto camera = data.engine->createCamera();
    camera->setProjection(VERTICAL_FOV,
                          float(data.width) / float(data.height),
                          NEAR_PLANE, MIN_FAR_PLANE,
                          filament::Camera::Fov::VERTICAL);
    view->setCamera(camera);

    data.scene = scene;
    data.view = view;
}
#endif // USE_FILAMENT

void OnResize(GLFWwindow *w, int osWidth, int osHeight) {

    float xscale, yscale;
    glfwGetWindowContentScale(w, &xscale, &yscale);
    WindowData &data = *(WindowData*)glfwGetWindowUserPointer(w);
    data.width = osWidth * xscale;
    data.height = osHeight * yscale;

    std::cout << "[debug] OnResize(" << osWidth << ", " << osHeight << "), pixel size: "
              << data.width << ", " << data.height << std::endl;

#if USE_FILAMENT
    CreateSwapChain(w);
    data.view->setViewport({0, 0, uint32_t(data.width), uint32_t(data.height)});

    std::cout << "[debug] set viewport: 0, 0, " << data.width << ", " << data.height << std::endl;

    SetRandomBGColor(data);
#else
    glfwMakeContextCurrent(w);
    glViewport(0, 0, data.width, data.height);
    float red = double(random()) / double(RAND_MAX);
    float green = double(random()) / double(RAND_MAX);
    float blue = double(random()) / double(RAND_MAX);
    glClearColor(red, green, blue, 1.0);
#endif // USE_FILAMENT
}

void OnKey(GLFWwindow *w, int key, int scancode, int action, int mods) {
#if USE_FILAMENT
    if (action == GLFW_PRESS) {
        SetRandomBGColor(*(WindowData*)glfwGetWindowUserPointer(w));
        open3d::gui::PostNativeExposeEvent(w);
    }
#endif // USE_FILAMENT
}

void OnDraw(GLFWwindow *w) {
    std::cout << "[debug] OnDraw" << std::endl;

#if USE_FILAMENT
    WindowData &data = *(WindowData*)glfwGetWindowUserPointer(w);
    data.renderer->beginFrame(data.swapChain);
    data.renderer->render(data.view);
    data.renderer->endFrame();
#else
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(w);
#endif // USE_FILAMENT
}

bool CreateWindow() {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);

    int width = 800;
    int height = 600;
    GLFWwindow* w = glfwCreateWindow(width, height, "Open3D - GLFWTest", NULL, NULL);
    if (!w) {
        std::cerr << "[ERROR] could not create window" << std::endl;
        return false;
    }

    gWindowData[w] = WindowData();
    WindowData &data = gWindowData[w];
    glfwSetWindowUserPointer(w, &data);
    float xscale, yscale;
    glfwGetWindowContentScale(w, &xscale, &yscale);
    data.width = width * xscale;
    data.height = height * yscale;

#if USE_FILAMENT
    filament::Engine *engine = filament::Engine::create(filament::backend::Backend::OPENGL);
    data.engine = engine;
    data.renderer = engine->createRenderer();
    CreateSwapChain(w);
    CreateFilamentScene(data);
#else
    glfwMakeContextCurrent(w);
    glClearColor(1, 1, 1, 1);
#endif // USE_FILAMENT

    glfwSetWindowRefreshCallback(w, OnDraw);
    glfwSetWindowSizeCallback(w, OnResize);
    glfwSetKeyCallback(w, OnKey);

    while (!glfwWindowShouldClose(w)) {
        glfwWaitEvents();
    }

#if USE_FILAMENT
    data.engine->destroy(data.swapChain);
    data.engine->destroy(data.scene);
    data.engine->destroy(data.view);
    data.engine->destroy(data.renderer);
    filament::Engine::destroy(data.engine);
#endif // USE_FILAMENT

    glfwDestroyWindow(w);
    return true;
}

int main(int argc, const char *argv[]) {
    if (!glfwInit()) {
        std::cerr << "[ERROR] could not initialize GLFW" << std::endl;
        return 1;
    }

    int major, minor, rev;
    glfwGetVersion(&major, &minor, &rev);
    std::cout << "GLFW version " << major << "." << minor << "." << rev << std::endl;

    CreateWindow();

    glfwTerminate();
}

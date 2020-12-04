include(ExternalProject)

ExternalProject_Add(
    ext_imgui
    PREFIX imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.79
    GIT_SHALLOW ON
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_imgui SOURCE_DIR)
set(IMGUI_SRC_DIR ${SOURCE_DIR})

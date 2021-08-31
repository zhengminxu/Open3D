# ---------------------------
# Open3D ISPC language module
# ---------------------------
#
# This module simulates CMake's first-class ISPC language support introduced in CMake 3.19.
# Use this module to bridge compatibility with unsupported generators, e.g. Visual Studio.
#
# Drop-in replacements:
# - open3d_ispc_enable_language()
# - open3d_ispc_add_library()
# - open3d_ispc_add_executable()
# - open3d_ispc_target_sources()
#
# Additional workaround functionality:
# - open3d_ispc_link_object_files()
#
# For a list of limitations, see the documentation of the individual functions.

# Internal helper function.
function(open3d_get_target_relative_object_dir target output_dir)
    unset(${output_dir})

    get_target_property(TARGET_BINARY_DIR ${target} BINARY_DIR)
    file(RELATIVE_PATH TARGET_RELATIVE_BINARY_DIR "${CMAKE_BINARY_DIR}" "${TARGET_BINARY_DIR}")

    set(${output_dir} "${TARGET_RELATIVE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${target}.dir" PARENT_SCOPE)
endfunction()

# Internal helper function.
function(open3d_init_target_property target property)
    if (DEFINED CMAKE_${property})
        set(property_value "${CMAKE_${property}}")
    elseif (${ARGC} EQUAL 3)
        set(property_value "${ARGV2}")
    endif()
    if (DEFINED property_value)
        set_target_properties(${target} PROPERTIES ${property} "${property_value}")
    endif()
endfunction()

# Internal helper function.
function(open3d_get_target_property output_variable target property default_value)
    unset(${output_variable})

    get_target_property(${output_variable} ${target} ${property})
    if (${output_variable} STREQUAL "${output_variable}-NOTFOUND")
        if (CMAKE_${property})
            set(${output_variable} "${CMAKE_${property}}")
        else()
            set(${output_variable} "${default_value}")
        endif()
    endif()

    set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
endfunction()

# Internal helper function.
function(open3d_evaluate_genex output_variable input_value genex keep_genex_content)
    unset(${output_variable})

    if ("${input_value}" MATCHES "^\\$<${genex}:(.+)>$")
        if (keep_genex_content)
            set(${output_variable} "${CMAKE_MATCH_1}")
        else()
            set(${output_variable} "")
        endif()
    else()
        set(${output_variable} "${input_value}")
    endif()

    set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
endfunction()

# Internal helper function.
function(open3d_collect_property_values target property accepted_genex_conditions output_variable print_all_props)
    unset(${output_variable})

    # Search target
    get_target_property(TARGET_PROPS ${target} ${property})

    # Concatenate generator expressions with lists
    unset(prop)
    while(TARGET_PROPS)
        list(POP_FRONT TARGET_PROPS PROP_PART)
        list(APPEND prop ${PROP_PART})

        # Check for partial generator expression
        if (NOT prop MATCHES "^\\$<.+:[^\\$<>]+[^>]$")
            # Body of the concatenated loop
            if (print_all_props)
                message(STATUS "Property of ${target}: ${prop}")
            endif()

            foreach(genex IN ITEMS ${accepted_genex_conditions})
                open3d_evaluate_genex(prop "${prop}" "${genex}" TRUE)
            endforeach()
            open3d_evaluate_genex(prop "${prop}" ".+" FALSE)

            if (prop)
                list(APPEND ${output_variable} "${prop}")
            endif()

            # Clean up loop
            unset(prop)
        endif()
    endwhile()

    # Search first-level/direct dependencies of target
    get_target_property(TARGET_LIBRARIES ${target} LINK_LIBRARIES)
    if (TARGET_LIBRARIES)
        foreach(lib IN ITEMS ${TARGET_LIBRARIES})
            if (TARGET ${lib})
                get_target_property(TARGET_PROPS ${lib} INTERFACE_${property})

                # Concatenate generator expressions with lists
                unset(prop)
                while(TARGET_PROPS)
                    list(POP_FRONT TARGET_PROPS PROP_PART)
                    list(APPEND prop ${PROP_PART})

                    # Check for partial generator expression
                    if (NOT prop MATCHES "^\\$<.+:[^\\$<>]+[^>]$")
                        # Body of the concatenated loop
                        if (print_all_props)
                            message(STATUS "Property of ${lib}: ${prop}")
                        endif()

                        foreach(genex IN LISTS accepted_genex_conditions)
                            open3d_evaluate_genex(prop "${prop}" "${genex}" TRUE)
                        endforeach()
                        open3d_evaluate_genex(prop "${prop}" ".+" FALSE)

                        if (prop)
                            list(APPEND ${output_variable} "${prop}")
                        endif()

                        # Clean up loop
                        unset(prop)
                    endif()
                endwhile()

            endif()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES ${output_variable})
    set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
endfunction()

# open3d_ispc_enable_language(<lang>)
#
# This is a drop-in replacement of enable_language(...).
#
# Finds the ISPC compiler via the ISPC environment variable or
# the CMAKE_ISPC_COMPILER variable and enables the ISPC language.
#
# The following variables will be defined:
# - CMAKE_ISPC_COMPILER
# - CMAKE_ISPC_COMPILER_ID
# - CMAKE_ISPC_COMPILER_VERSION
#
# Limitations:
# - Only ISPC compiler with compiler ID "Intel" is supported.
# - Other language-related variables are not defined.
# - Can only be used with ISPC argument
macro(open3d_ispc_enable_language lang)
    # Check correct usage
    if (NOT "${lang}" STREQUAL "ISPC")
        message(FATAL_ERROR "Enabling language \"${lang}\" != \"ISPC\" is not possible. Only \"open3d_ispc_enable_language(ISPC)\" is supported")
    endif()

    if(NOT ISPC_FORCE_LEGACY AND (CMAKE_GENERATOR MATCHES "Make" OR CMAKE_GENERATOR MATCHES "Ninja"))
        enable_language(ISPC)
    else()
        # Set CMAKE_ISPC_COMPILER
        get_filename_component(ISPC_ENV_DIR $ENV{ISPC} DIRECTORY)
        find_program(CMAKE_ISPC_COMPILER REQUIRED
            NAMES ispc
            PATHS ${ISPC_ENV_DIR}
            NO_DEFAULT_PATH
        )

        # Set CMAKE_ISPC_COMPILER_ID
        set(CMAKE_ISPC_COMPILER_ID "Intel")

        # Set CMAKE_ISPC_COMPILER_VERSION
        # This also tests if the compiler can be invoked on this platform.
        execute_process(
            COMMAND ${CMAKE_ISPC_COMPILER} --version
            OUTPUT_VARIABLE output
            RESULT_VARIABLE result
        )
        if (result AND NOT result EQUAL 0)
            message(FATAL_ERROR "Testing ISPC compiler ${CMAKE_ISPC_COMPILER} failed. The compiler might be broken.")
        else()
            if (output MATCHES [[ISPC\), ([0-9]+\.[0-9]+(\.[0-9]+)?)]])
                set(CMAKE_ISPC_COMPILER_VERSION "${CMAKE_MATCH_1}")
            else()
                message(WARNING "Unknown ISPC compiler version.")
            endif()
        endif()
        message(STATUS "Found ISPC compiler: ${CMAKE_ISPC_COMPILER_ID} ${CMAKE_ISPC_COMPILER_VERSION} (${CMAKE_ISPC_COMPILER})")
    endif()
endmacro()

# open3d_ispc_add_library(...)
#
# This is a drop-in replacement of add_library(..).
#
# Forwards all arguments to add_library(...) and sets up ISPC properties.
#
# Limitations:
# - Adding ISPC source files is only supported via open3d_ispc_target_sources(...).
macro(open3d_ispc_add_library)
    # Check correct usage
    foreach(arg IN ITEMS ${ARGV})
        get_filename_component(FILE_EXT "${arg}" LAST_EXT)

        if (FILE_EXT STREQUAL ".ispc")
            message(FATAL_ERROR "Passing ISPC file \"${arg}\" via \"open3d_ispc_add_library()\" is not supported. Use \"open3d_ispc_target_sources()\" instead.")
        endif()
    endforeach()

    add_library(${ARGV})

    if(NOT ISPC_FORCE_LEGACY AND (CMAKE_GENERATOR MATCHES "Make" OR CMAKE_GENERATOR MATCHES "Ninja"))
        # Nothing to do
    else()
        open3d_init_target_property(${ARGV0} ISPC_HEADER_SUFFIX "_ispc.h")
        open3d_init_target_property(${ARGV0} ISPC_HEADER_DIRECTORY)
        open3d_init_target_property(${ARGV0} ISPC_INSTRUCTION_SETS)
    endif()
endmacro()

# open3d_ispc_add_executable(...)
#
# This is a drop-in replacement of add_executable(..).
#
# Forwards all arguments to add_executable(...) and sets up ISPC properties.
#
# Limitations:
# - Adding ISPC source files is only supported via open3d_ispc_target_sources(...).
macro(open3d_ispc_add_executable)
    # Check correct usage
    foreach(arg IN ITEMS ${ARGV})
        get_filename_component(FILE_EXT "${arg}" LAST_EXT)

        if (FILE_EXT STREQUAL ".ispc")
            message(FATAL_ERROR "Passing ISPC file \"${arg}\" via \"open3d_ispc_add_executable()\" is not supported. Use \"open3d_ispc_target_sources()\" instead.")
        endif()
    endforeach()

    add_executable(${ARGV})

    if(NOT ISPC_FORCE_LEGACY AND (CMAKE_GENERATOR MATCHES "Make" OR CMAKE_GENERATOR MATCHES "Ninja"))
        # Nothing to do
    else()
        open3d_init_target_property(${ARGV0} ISPC_HEADER_SUFFIX "_ispc.h")
        open3d_init_target_property(${ARGV0} ISPC_HEADER_DIRECTORY)
        open3d_init_target_property(${ARGV0} ISPC_INSTRUCTION_SETS)
    endif()
endmacro()

# open3d_ispc_target_sources(<target>
#     PRIVATE <src1> [<src2>...]
# )
#
# This is a drop-in replacement of target_sources(...).
#
# Forwards any non-ISPC source files to the target_sources() command and
# adds custom build rules for ISPC source files.
#
# Limitations:
# - Only PRIVATE sources are supported.
# - Properties that affect build rule generation must be specified before
#   calling this function. This includes:
#   - (CMAKE_)ISPC_OUTPUT_EXTENSION
#   - (CMAKE_)ISPC_HEADER_DIRECTORY
#   - (CMAKE_)ISPC_HEADER_SUFFIX
#   - (CMAKE_)ISPC_FLAGS
#   - INCLUDE_DIRECTORIES
#   - COMPILE_DEFINITIONS
#   - COMPILE_OPTIONS
# - Dependency scanning for .isph header files is limited by the IMPLICIT_DEPENDS option of add_custom_command.
function(open3d_ispc_target_sources target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "PRIVATE")

    # Check correct usage
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if (ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for arguments: ${ARG_KEYWORDS_MISSING_VALUES}")
    endif()

    if (NOT ARG_PRIVATE)
        message(FATAL_ERROR "No sources specified.")
    endif()

    if(NOT ISPC_FORCE_LEGACY AND (CMAKE_GENERATOR MATCHES "Make" OR CMAKE_GENERATOR MATCHES "Ninja"))
        target_sources(${target} PRIVATE
            ${ARG_PRIVATE}
        )
    else()
        open3d_get_target_relative_object_dir(${target} TARGET_RELATIVE_OBJECT_DIR)

        # Use object file extension from C language
        if (NOT DEFINED CMAKE_ISPC_OUTPUT_EXTENSION)
            set(CMAKE_ISPC_OUTPUT_EXTENSION ${CMAKE_C_OUTPUT_EXTENSION})
        endif()

        open3d_get_target_property(TARGET_HEADER_SUFFIX ${target} ISPC_HEADER_SUFFIX "_ispc.h")
        open3d_get_target_property(TARGET_HEADER_DIRECTORY ${target} ISPC_HEADER_DIRECTORY "${CMAKE_BINARY_DIR}/${TARGET_RELATIVE_OBJECT_DIR}")
        open3d_get_target_property(TARGET_INSTRUCTION_SETS ${target} ISPC_INSTRUCTION_SETS "")

        # Use RelWithDebInfo flags
        if (NOT DEFINED CMAKE_ISPC_FLAGS)
            set(CMAKE_ISPC_FLAGS -O2 -g -DNDEBUG)
        endif()

        # Set PIC flag
        open3d_get_target_property(TARGET_POSITION_INDEPENDENT_CODE ${target} POSITION_INDEPENDENT_CODE "")
        if (TARGET_POSITION_INDEPENDENT_CODE)
            set(TARGET_PIC_FLAG --pic)
        endif()

        # Set ISA flag
        if (TARGET_INSTRUCTION_SETS)
            # Detect ISA suffixes
            list(LENGTH TARGET_INSTRUCTION_SETS TARGET_INSTRUCTION_SETS_LENGTH)
            if (TARGET_INSTRUCTION_SETS_LENGTH GREATER 1)
                foreach(isa IN LISTS TARGET_INSTRUCTION_SETS)
                    if (isa MATCHES "^([a-z0-9]+)\-i[0-9]+x[0-9]+$")
                        # Special case handling for AVX
                        if (CMAKE_MATCH_1 STREQUAL "avx1")
                            list(APPEND TARGET_INSTRUCTION_SET_SUFFIXES "_avx")
                        else()
                            list(APPEND TARGET_INSTRUCTION_SET_SUFFIXES "_${CMAKE_MATCH_1}")
                        endif()
                    else()
                        message(WARNING "Could not find suffix of ISPC instruction set \"${isa}\". This may lead to compilation or linker errors.")
                    endif()
                endforeach()
                list(REMOVE_DUPLICATES TARGET_INSTRUCTION_SET_SUFFIXES)
            endif()

            string(REPLACE ";" "," TARGET_INSTRUCTION_SETS_FLAG "${TARGET_INSTRUCTION_SETS}")
            set(TARGET_ISA_FLAGS --target=${TARGET_INSTRUCTION_SETS_FLAG})
        endif()

        # Make header files discoverable in C++ code
        target_include_directories(${target} PRIVATE
            ${TARGET_HEADER_DIRECTORY}
        )

        # Collect build flags
        set(ACCEPTED_GENERATOR_EXPRESSION_CONDITIONS
            "BUILD_INTERFACE"
            "\\$<COMPILE_LANGUAGE:ISPC>"
            "\\$<COMPILE_LANG_AND_ID:ISPC,.+>"
        )
        open3d_collect_property_values(${target} INCLUDE_DIRECTORIES "${ACCEPTED_GENERATOR_EXPRESSION_CONDITIONS}" OUTPUT_TARGET_INCLUDES FALSE)
        open3d_collect_property_values(${target} COMPILE_DEFINITIONS "${ACCEPTED_GENERATOR_EXPRESSION_CONDITIONS}" OUTPUT_TARGET_DEFINITIONS FALSE)
        open3d_collect_property_values(${target} COMPILE_OPTIONS "${ACCEPTED_GENERATOR_EXPRESSION_CONDITIONS}" OUTPUT_TARGET_OPTIONS FALSE)

        list(TRANSFORM OUTPUT_TARGET_INCLUDES PREPEND "-I")
        list(TRANSFORM OUTPUT_TARGET_DEFINITIONS PREPEND "-D")

        foreach (file IN LISTS ARG_PRIVATE)
            get_filename_component(FILE_EXT "${file}" LAST_EXT)

            if (NOT FILE_EXT STREQUAL ".ispc")
                # Forward non-ISPC files
                target_sources(${target} PRIVATE ${file})
            else()
                # Process ISPC files
                get_filename_component(FILE_FULL_PATH "${file}" ABSOLUTE)

                get_target_property(TARGET_SOURCE_DIR ${target} SOURCE_DIR)
                file(RELATIVE_PATH FILE_RELATIVE_PATH "${TARGET_SOURCE_DIR}" "${FILE_FULL_PATH}")

                get_filename_component(FILE_NAME "${file}" NAME_WE)

                set(HEADER_FILE_FULL_PATH "${TARGET_HEADER_DIRECTORY}/${FILE_NAME}${TARGET_HEADER_SUFFIX}")

                set(OBJECT_FILE_RELATIVE_PATH "${TARGET_RELATIVE_OBJECT_DIR}/${FILE_RELATIVE_PATH}${CMAKE_ISPC_OUTPUT_EXTENSION}")
                set(OBJECT_FILE_FULL_PATH "${CMAKE_BINARY_DIR}/${OBJECT_FILE_RELATIVE_PATH}")

                # Determine expected object and header files
                set(OBJECT_FILE_LIST ${OBJECT_FILE_FULL_PATH})
                set(HEADER_FILE_LIST ${HEADER_FILE_FULL_PATH})
                foreach(suffix IN LISTS TARGET_INSTRUCTION_SET_SUFFIXES)
                    # Per-ISA header files
                    if (TARGET_HEADER_SUFFIX MATCHES "^([A-Za-z0-9_]*)(\.[A-Za-z0-9_]*)$")
                        set(TARGET_INSTRUCTION_SET_HEADER_SUFFIX "${CMAKE_MATCH_1}${suffix}${CMAKE_MATCH_2}")

                        list(APPEND HEADER_FILE_LIST "${TARGET_HEADER_DIRECTORY}/${FILE_NAME}${TARGET_INSTRUCTION_SET_HEADER_SUFFIX}")
                    else()
                        message(WARNING "Could not generate per-ISA header suffixes from \"${TARGET_HEADER_SUFFIX}\".")
                    endif()

                    # Per-ISA object files
                    list(APPEND OBJECT_FILE_LIST "${CMAKE_BINARY_DIR}/${TARGET_RELATIVE_OBJECT_DIR}/${FILE_RELATIVE_PATH}${suffix}${CMAKE_ISPC_OUTPUT_EXTENSION}")
                endforeach()

                # Note:
                # Passing -MMM <depfile> to the ISPC compiler allows for generating dependency files.
                # However, they are not correctly recognized. Use IMPLICIT_DEPENDS instead.
                add_custom_command(
                    OUTPUT ${OBJECT_FILE_LIST} ${HEADER_FILE_LIST}
                    COMMAND ${CMAKE_ISPC_COMPILER} ${OUTPUT_TARGET_DEFINITIONS} ${OUTPUT_TARGET_INCLUDES} ${CMAKE_ISPC_FLAGS} ${TARGET_ISA_FLAGS} ${TARGET_PIC_FLAG} ${OUTPUT_TARGET_OPTIONS} -o ${OBJECT_FILE_FULL_PATH} --emit-obj ${FILE_FULL_PATH} -h ${HEADER_FILE_FULL_PATH}
                    IMPLICIT_DEPENDS C ${FILE_FULL_PATH}
                    COMMENT "Building ISPC object ${OBJECT_FILE_RELATIVE_PATH}"
                    MAIN_DEPENDENCY ${FILE_FULL_PATH} DEPENDS ${CMAKE_ISPC_COMPILER}
                    VERBATIM
                )

                if (ISPC_PRINT_LEGACY_COMPILE_COMMANDS)
                    # Simulate internal post-processing of lists
                    string(REPLACE ";" " " CMAKE_ISPC_FLAGS_PROCESSED "${CMAKE_ISPC_FLAGS}")
                    string(REPLACE ";" " " OUTPUT_TARGET_DEFINITIONS_PROCESSED "${OUTPUT_TARGET_DEFINITIONS}")
                    string(REPLACE ";" " " OUTPUT_TARGET_INCLUDES_PROCESSED "${OUTPUT_TARGET_INCLUDES}")
                    string(REPLACE ";" " " OUTPUT_TARGET_OPTIONS_PROCESSED "${OUTPUT_TARGET_OPTIONS}")

                    set(FILE_COMPILE_COMMAND_PROCESSED "${CMAKE_ISPC_COMPILER} ${OUTPUT_TARGET_DEFINITIONS_PROCESSED} ${OUTPUT_TARGET_INCLUDES_PROCESSED} ${CMAKE_ISPC_FLAGS_PROCESSED} ${TARGET_ISA_FLAGS} ${TARGET_PIC_FLAG} ${OUTPUT_TARGET_OPTIONS_PROCESSED} -o ${OBJECT_FILE_FULL_PATH} --emit-obj ${FILE_FULL_PATH} -h ${HEADER_FILE_FULL_PATH}")

                    # Strip double spaces caused by empty lists
                    string(REGEX REPLACE "[ ]+" " " FILE_COMPILE_COMMAND_PROCESSED "${FILE_COMPILE_COMMAND_PROCESSED}")

                    message(STATUS "${file}: ${FILE_COMPILE_COMMAND_PROCESSED}")
                endif()

                list(APPEND TARGET_OBJECT_FILES "${OBJECT_FILE_LIST}")

                # Add dependency of ISPC object file to target.
                # NOTE: If <target> is an object library, this does not make
                #       the file appear in the list $<TARGET_OBJECTS:${target}>.
                target_sources(${target} PRIVATE
                    ${OBJECT_FILE_LIST}
                )
            endif()
        endforeach()

        # Add files to GENERATED_OBJECT_FILES property.
        # This will later be used to resolve library.
        get_target_property(TARGET_GENERATED_OBJECT_FILES ${target} GENERATED_OBJECT_FILES)
        if (NOT TARGET_GENERATED_OBJECT_FILES)
            set(TARGET_GENERATED_OBJECT_FILES "")
        endif()
        list(APPEND TARGET_GENERATED_OBJECT_FILES ${TARGET_OBJECT_FILES})
        set_target_properties(${target} PROPERTIES GENERATED_OBJECT_FILES "${TARGET_GENERATED_OBJECT_FILES}")
    endif()
endfunction()

# open3d_ispc_link_object_files(<target>
#     PRIVATE <dep1> [<dep2>...]
# )
#
# Links the generated object files of the object libraries <dep1> [<dep2>...]
# into <target>.
#
# Since $<TARGET_OBJECTS:<target>> does not include manually generated object
# files for any object library <target>, this step is required to ensure
# correct linking.
function(open3d_ispc_link_object_files target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "PRIVATE")

    # Check correct usage
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if (ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for arguments: ${ARG_KEYWORDS_MISSING_VALUES}")
    endif()

    if (NOT ARG_PRIVATE)
        message(FATAL_ERROR "No dependencies specified.")
    endif()

    if(NOT ISPC_FORCE_LEGACY AND (CMAKE_GENERATOR MATCHES "Make" OR CMAKE_GENERATOR MATCHES "Ninja"))
        # Nothing to do
    else()
        # Process dependencies
        foreach (dep IN LISTS ARG_PRIVATE)
            get_target_property(DEP_GENERATED_OBJECT_FILES ${dep} GENERATED_OBJECT_FILES)
            if (DEP_GENERATED_OBJECT_FILES STREQUAL "DEP_GENERATED_OBJECT_FILES-NOTFOUND")
                set(DEP_GENERATED_OBJECT_FILES "")
            endif()

            if (NOT DEP_GENERATED_OBJECT_FILES STREQUAL "")
                target_link_libraries(${target} PRIVATE ${DEP_GENERATED_OBJECT_FILES})
            endif()
        endforeach()
    endif()
endfunction()

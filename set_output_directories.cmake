function(set_output_directories)
    # Did we get any optional args?
    set(optional_argv ${ARGV})
    list(LENGTH optional_argv optional_argc)
    if (${optional_argc} GREATER 0)
        list(GET optional_argv 0 custom_runtime_output_dir)
    endif()

    # Set the base output directories
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin PARENT_SCOPE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib PARENT_SCOPE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib PARENT_SCOPE)

    # Set the output directories for specific configurations
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_BINARY_DIR}/bin/${OUTPUTCONFIG} PARENT_SCOPE)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_BINARY_DIR}/lib/${OUTPUTCONFIG} PARENT_SCOPE)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_BINARY_DIR}/lib/${OUTPUTCONFIG} PARENT_SCOPE)
    endforeach()

    # Modify binary runtime output directory with given path if provided
    if(DEFINED custom_runtime_output_dir)
        message("Setting runtime output directory to ${custom_runtime_output_dir}")
        # Set the output directories for specific configurations
        foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${custom_runtime_output_dir} PARENT_SCOPE)
            set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${custom_runtime_output_dir} PARENT_SCOPE)
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${custom_runtime_output_dir} PARENT_SCOPE)
        endforeach()
    endif()
endfunction()

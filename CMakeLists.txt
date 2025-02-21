cmake_minimum_required(VERSION 3.16)
project(delta_vins)

if(NOT DEFINED USE_ROS)
    set(USE_ROS TRUE)
endif()

if(USE_ROS)
    add_definitions(-DUSE_ROS2=1)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(USE_ROS)
    # find dependencies
    find_package(ament_cmake REQUIRED)
    find_package(OpenCV REQUIRED)
    find_package(cv_bridge REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(rosbag2_cpp REQUIRED)
    # uncomment the following section in order to fill in
    # further dependencies manually.
    # find_package(<dependency> REQUIRED)
else() # conan build
    find_package(OpenCV REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(Pangolin REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)


if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
    set(ARM TRUE)
    message("Architecture ARM")
    add_definitions("-DPLATFORM_ARM")
endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(ARM TRUE)
    message("Architecture ARM")
    add_definitions("-DPLATFORM_ARM")
endif()

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Windows)
    set(Windows TRUE)
    message("OS:Windows")
endif()

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Linux)
    set(Linux TRUE)
    message(OS : Linux)

endif()

set(BUILD_TESTS TRUE)

set(source_root ${PROJECT_SOURCE_DIR})
set(src_dir ${source_root}/src)

message("source_root: ${source_root}")
message("src_dir: ${src_dir}")

if(NOT USE_ROS)
    if(ENABLE_VISUALIZER_TCP)

    add_definitions(-DENABLE_VISUALIZER_TCP=1)

    elseif(ENABLE_VISUALIZER)

    add_definitions(-DENABLE_VISUALIZER=1)
    endif()

endif()

add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=0) # disable stack allocation limit

include_directories(include/framework)
include_directories(3rdParty/CmdParser)

if(NOT USE_ROS)
    add_library(SlamVisualizer SHARED
        ${source_root}/SlamVisualizer/SlamVisualizer.cpp
    )

    target_link_libraries(SlamVisualizer
        Eigen3::Eigen
        opencv::opencv
        pangolin::pangolin
    )
endif()

file(GLOB src_files
    ${src_dir}/*.cpp
    ${src_dir}/Algorithm/*.cpp
    ${src_dir}/Algorithm/DataAssociation/*.cpp
    ${src_dir}/Algorithm/IMU/*.cpp
    ${src_dir}/Algorithm/vision/*.cpp
    ${src_dir}/Algorithm/vision/camModel/*.cpp
    ${src_dir}/Algorithm/vision/fast/*.cpp
    ${src_dir}/dataStructure/*.cpp
    ${src_dir}/utils/*.cpp
    ${src_dir}/framework/*.cpp
    ${src_dir}/IO/dataBuffer/*.cpp
    ${src_dir}/IO/dataSource/*.cpp
    ${src_dir}/IO/dataOutput/*.cpp
    ${src_dir}/Algorithm/solver/*.cpp
    ${src_dir}/Algorithm/Initializer/*.cpp
)

include_directories(
    ${source_root}/include
    ${source_root}/3rdParty
    ${source_root}/3rdParty/serial/include
    ${source_root}/SlamVisualizer
)


add_library(${CMAKE_PROJECT_NAME} SHARED
    ${src_files}
)

if(USE_ROS)
    ament_target_dependencies(${CMAKE_PROJECT_NAME}
        "rclcpp"
        "sensor_msgs"
        "std_msgs"
        "cv_bridge"
        "nav_msgs"
        "rosbag2_cpp"
    )
    target_link_libraries(${CMAKE_PROJECT_NAME}
        Eigen3::Eigen
        ${OpenCV_LIBS}
        # pangolin::pangolin
    )
else()

    target_link_libraries(${CMAKE_PROJECT_NAME}
        Eigen3::Eigen
        opencv::opencv
        pangolin::pangolin
    )

endif()


target_precompile_headers(${CMAKE_PROJECT_NAME} PRIVATE include/precompile.h)



if(UNIX)
    set(LINK_LIBS
        ${CMAKE_PROJECT_NAME}
        pthread
    )
    if(NOT USE_ROS)
        set(VIS_LIB
            GL
            GLEW
        )
    endif()
endif()

if(USE_ROS)
    set(LINK_LIBS
        ${LINK_LIBS}
    )
else()
    set(LINK_LIBS
        ${LINK_LIBS}
        SlamVisualizer
        ${VIS_LIB}
    )
endif()


add_executable(RunDeltaVINS
    ${source_root}/examples/RunDeltaVINS.cpp
)

target_link_libraries(
    RunDeltaVINS
    ${LINK_LIBS}
)


set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PUBLIC_HEADER
    "include/framework/slamAPI.h;include/dataStructure/sensorStructure.h"
)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PRIVATE_HEADER
    "include/utils/basicTypes.h")

if(USE_ROS)
    ament_export_libraries(${CMAKE_PROJECT_NAME})

    set(INSTALL_TARGETS
        ${CMAKE_PROJECT_NAME} RunDeltaVINS
    )

    if(BUILD_TESTING)
        find_package(ament_lint_auto REQUIRED)
        # the following line skips the linter which checks for copyrights
        # comment the line when a copyright and license is added to all source files
        set(ament_cmake_copyright_FOUND TRUE)
        # the following line skips cpplint (only works in a git repo)
        # comment the line when this package is in a git repo and when
        # a copyright and license is added to all source files
        set(ament_cmake_cpplint_FOUND TRUE)
        ament_lint_auto_find_test_dependencies()
        add_subdirectory(test)
    endif()

    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    # Install launch files.
    install(DIRECTORY
        launch rviz Config
        DESTINATION share/${PROJECT_NAME}/
    )
    install(
        TARGETS RunDeltaVINS ${CMAKE_PROJECT_NAME}
        DESTINATION lib/${PROJECT_NAME})
    ament_package()

endif()

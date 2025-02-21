# add test_triangulation
add_executable(test_triangulation test_triangulation.cpp)

target_link_libraries(test_triangulation
    ${LINK_LIBS}
    gtest
)


install(TARGETS test_triangulation
    DESTINATION lib/${PROJECT_NAME})

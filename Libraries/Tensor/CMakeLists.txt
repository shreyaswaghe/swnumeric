# Link other libraries here if needed target_link_libraries(libnumeric_vector
# other_library)

add_executable(test_vector ${CMAKE_CURRENT_SOURCE_DIR}/tests/testVector.cpp)
target_link_libraries(test_vector PRIVATE swnumeric)
add_test(NAME VectorTests COMMAND test_vector)

# add_executable(test_matrix ${CMAKE_CURRENT_SOURCE_DIR}/tests/testMatrix.cpp)
# target_link_libraries(test_matrix PRIVATE swnumeric) add_test(NAME MatrixTests
# COMMAND test_matrix)

add_executable(test_tensor ${CMAKE_CURRENT_SOURCE_DIR}/tests/testTensor.cpp)
target_link_libraries(test_tensor PRIVATE swnumeric)
add_test(NAME TensorTests COMMAND test_tensor)

# add_executable(test_matrix ${CMAKE_CURRENT_SOURCE_DIR}/tests/testMatrix.cpp)
# target_link_libraries(test_matrix PRIVATE swnumeric) add_test(NAME MatrixTests
# COMMAND test_matrix)

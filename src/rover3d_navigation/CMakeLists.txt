cmake_minimum_required(VERSION 3.8)
project(rover3d_navigation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python library from include/rover3d_navigation
install(DIRECTORY include/rover3d_navigation/
  DESTINATION lib/python3.10/site-packages/rover3d_navigation
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Node executables from src/
install(PROGRAMS
  src/navigator.py
  src/planning_apf_node.py
  src/gmm_goal_publisher_node.py
  scripts/precompute_config_prior.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES package.xml README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

# Meson overlay for glm (header only), applied by external/glm.wrap
project('glm', 'cpp', version : '1.0.3', license : 'MIT')
glm_dep = declare_dependency(include_directories : include_directories('.', is_system : true))
meson.override_dependency('glm', glm_dep)

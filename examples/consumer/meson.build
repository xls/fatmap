# Consuming a fatmap release package from Meson (pkg-config, or CMake as a
# fallback lookup method when pkg-config is not installed):
#   meson setup build --pkg-config-path /path/to/fatmap-<version>-<platform>/lib/pkgconfig \
#                     --cmake-prefix-path /path/to/fatmap-<version>-<platform>
#   meson compile -C build
# (to build fatmap from source as a Meson subproject instead, see
# docs/PACKAGING.md)
project('fatmap_consumer', ['c', 'cpp'], default_options : ['c_std=c11', 'cpp_std=c++17'])

fatmap = dependency('fatmap', version : '>=0.1', method : 'pkg-config', required : false)
if not fatmap.found()
  fatmap = dependency('fatmap', version : '>=0.1', method : 'cmake', modules : ['fatmap::fatmap'])
endif
# the C++ wrapper is header only and installed next to the C headers
fatmapxx = fatmap

executable('consumer_c', 'main.c', dependencies : fatmap)
executable('consumer_cpp', 'main.cpp', dependencies : fatmapxx)

project('fatmap', ['c', 'cpp'],
  version : '0.6.0',
  license : 'MIT',
  meson_version : '>=1.3.0',
  subproject_dir : 'external',
  default_options : [
    'c_std=c11',
    'cpp_std=c++17',
    'warning_level=2',
    'buildtype=release',
    'default_library=static',
    'b_ndebug=if-release',
    'force_fallback_for=sdl3,glm,tracy',
  ])

fs = import('fs')
cc = meson.get_compiler('c')
cpu = host_machine.cpu_family()
msvc_like = cc.get_argument_syntax() == 'msvc'

inc = include_directories('include')
subdir('include/fatmap') # fm_config.h (feature flags)

# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------
common_args = []
if msvc_like
  common_args += ['/D_CRT_SECURE_NO_WARNINGS', '/fp:precise']
else
  # no FMA contraction: SIMD backends must stay bit-identical to scalar
  common_args += cc.get_supported_arguments(['-ffp-contract=off', '-fno-math-errno'])
endif

profile_args = ['-DFM_PROFILE=' + (get_option('profile') ? '1' : '0')]
pub_args = get_option('default_library') == 'static' ? [] : ['-DFM_SHARED']

m_dep = cc.find_library('m', required : false)

# threads are optional (-Dthreads=disabled for platforms without a thread
# library): command lists then execute serially on the calling thread
thread_dep = dependency('threads', required : get_option('threads'))
threads_on = thread_dep.found() and not get_option('threads').disabled()
thread_args = threads_on ? [] : ['-DFM_NO_THREADS']
# the Windows pool uses Win32 threads directly (no winpthread dependency)
thread_deps = threads_on and host_machine.system() != 'windows' ? [thread_dep] : []

# ---------------------------------------------------------------------------
# optional Tracy (external/tracy.wrap -> git clone of upstream, has its own meson)
# ---------------------------------------------------------------------------
tracy_deps = []
if get_option('tracy')
  tracy_deps += dependency('tracy', default_options : ['tracy_enable=true', 'warning_level=0'])
  tracy_deps += declare_dependency(compile_args : ['-DFM_TRACY=1'])
endif

# ---------------------------------------------------------------------------
# SIMD backends: each compiled with its own ISA flags, selected at runtime
# ---------------------------------------------------------------------------
simd_libs = []
simd_defs = []
if cpu in ['x86', 'x86_64']
  sse2_args = msvc_like ? [] : ['-msse2']
  avx2_args = msvc_like ? ['/arch:AVX2'] : ['-mavx2']
  simd_libs += static_library('fm_sse2', 'src/core/fm_kernels_sse2.c',
    c_args : common_args + sse2_args + pub_args, include_directories : inc, pic : true)
  simd_libs += static_library('fm_avx2', 'src/core/fm_kernels_avx2.c', 'src/3d/fm3d_spirv_avx2.c',
    c_args : common_args + avx2_args + pub_args, include_directories : inc, pic : true)
  simd_defs += ['-DFM_HAVE_SSE2', '-DFM_HAVE_AVX2']
  # AVX-512 (F / DQ / BW / VL): the SPIR-V executor at 16 lanes per vector
  avx512_args = msvc_like ? ['/arch:AVX512'] : ['-mavx512f', '-mavx512dq', '-mavx512bw', '-mavx512vl']
  if cc.has_multi_arguments(avx512_args)
    simd_libs += static_library('fm_avx512', 'src/3d/fm3d_spirv_avx512.c',
      c_args : common_args + avx512_args + pub_args, include_directories : inc, pic : true)
    simd_defs += ['-DFM_HAVE_AVX512_SPIRV']
  endif
elif cpu == 'aarch64'
  simd_libs += static_library('fm_neon', 'src/core/fm_kernels_neon.c',
    c_args : common_args + pub_args, include_directories : inc, pic : true)
  simd_defs += ['-DFM_HAVE_NEON']
endif

# ---------------------------------------------------------------------------
# libfatmap (C11)
# ---------------------------------------------------------------------------
core_src = files(
  'src/core/fm_cpu.c',
  'src/core/fm_color.c',
  'src/core/fm_surface.c',
  'src/core/fm_profile.c',
  'src/core/fm_blend_complex.c',
  'src/core/fm_kernels_scalar.c',
  'src/core/fm_raster.c',
  'src/core/fm_pipe.c',
  'src/core/fm_thread.c',
  'src/core/fm_cmdlist.c',
  'src/core/fm_arena.c',
)
twod_src = files(
  'src/2d/fm2d_path.c',
  'src/2d/fm2d_stroke.c',
  'src/2d/fm2d_ctx.c',
)
threed_src = files(
  'src/3d/fm3d_texture.c',
  'src/3d/fm3d_raster.c',
  'src/3d/fm3d_ctx.c',
  'src/3d/fm3d_spirv.c',
  'src/3d/fm3d_spirv_c.c',
  'src/3d/fm3d_spirv_inline.c',
)

is_sub = meson.is_subproject()

# static archives use the platform's native name (fatmap.lib with MSVC,
# libfatmap.a elsewhere) so they drop into any build system
lib_name_args = {}
if msvc_like and get_option('default_library') == 'static'
  lib_name_args = {'name_prefix' : '', 'name_suffix' : 'lib'}
endif

fatmap_lib = library('fatmap', core_src + twod_src + threed_src,
  kwargs : lib_name_args,
  c_args : common_args + simd_defs + profile_args + pub_args + thread_args + ['-DFM_BUILD'],
  include_directories : inc,
  link_whole : simd_libs,
  dependencies : [m_dep] + thread_deps + tracy_deps,
  gnu_symbol_visibility : 'hidden',
  install : not is_sub)

fatmap_dep = declare_dependency(
  link_with : fatmap_lib,
  include_directories : inc,
  compile_args : pub_args + profile_args,
  dependencies : [m_dep] + thread_deps + tracy_deps)
meson.override_dependency('fatmap', fatmap_dep)

if not is_sub
  install_subdir('include/fatmap', install_dir : get_option('includedir'),
    exclude_files : ['meson.build', 'fm_config.h.in'])
endif

# fm-spirvc: SPIR-V -> C (ahead of time shader compilation)
if get_option('shaders') and get_option('spirv') and not is_sub
  executable('fm-spirvc', 'tools/fm_spirvc.c',
    c_args : common_args, dependencies : fatmap_dep, install : true)
endif

# ---------------------------------------------------------------------------
# fatmap++ (C++17 header library wrapping 2D, later 3D)
# ---------------------------------------------------------------------------
fatmapxx_dep = declare_dependency(
  include_directories : include_directories('cpp/include'),
  dependencies : fatmap_dep)
meson.override_dependency('fatmapxx', fatmapxx_dep)

# ---------------------------------------------------------------------------
# consumer packages (installed next to the library, see docs/PACKAGING.md):
# pkg-config (Make, Meson, autotools, ...) and a CMake config package, so
# `find_package(fatmap)` works without fatmap itself being built with CMake
# ---------------------------------------------------------------------------
if not is_sub
  install_subdir('cpp/include/fatmap', install_dir : get_option('includedir'))

  # a static archive does not carry its dependencies, so they go in Libs
  # (not Libs.private) for plain `pkg-config --libs fatmap`
  pc_libs = [fatmap_lib]
  if m_dep.found()
    pc_libs += m_dep
  endif
  pc_libs += thread_deps
  pkg = import('pkgconfig')
  pkg.generate(
    name : 'fatmap',
    description : 'SIMD accelerated 2D (HTML canvas) and 3D software rasterizer',
    url : 'https://github.com/xls/fatmap',
    libraries : pc_libs,
    extra_cflags : pub_args)
  pkg.generate(
    name : 'fatmapxx',
    filebase : 'fatmapxx',
    description : 'C++17 header wrapper for fatmap',
    url : 'https://github.com/xls/fatmap',
    requires : 'fatmap')

  cmake_dir = get_option('libdir') / 'cmake' / 'fatmap'
  # relative path from <libdir>/cmake/fatmap back to the prefix (relocatable)
  cmake_up = '../..'
  foreach part : get_option('libdir').split('/')
    cmake_up += '/..'
  endforeach
  cmake_conf = configuration_data({
    'PREFIX_FROM_CONFIG' : cmake_up,
    'VERSION' : meson.project_version(),
    'LIBFILE' : fs.name(fatmap_lib.full_path()),
    'LIBDIR' : get_option('libdir'),
    'INCLUDEDIR' : get_option('includedir'),
    'SHARED' : get_option('default_library') == 'static' ? 'OFF' : 'ON',
    'NEED_LIBM' : m_dep.found() ? 'ON' : 'OFF',
    'NEED_THREADS' : thread_deps.length() > 0 ? 'ON' : 'OFF',
    'THREADS' : threads_on ? 'ON' : 'OFF',
    'PROFILE' : get_option('profile') ? 'ON' : 'OFF',
    'SIZEOF_VOID_P' : cc.sizeof('void*'),
  })
  configure_file(input : 'packaging/fatmapConfig.cmake.in', output : 'fatmapConfig.cmake',
    configuration : cmake_conf, install_dir : cmake_dir)
  configure_file(input : 'packaging/fatmapConfigVersion.cmake.in', output : 'fatmapConfigVersion.cmake',
    configuration : cmake_conf, install_dir : cmake_dir)
endif

# glm (external/glm.wrap -> git clone, header only). Used by the C++ tests
# for layout/value compatibility with fm_math.h; the C library never needs it.
glm_dep = is_sub ? dependency('', required : false) : dependency('glm', required : false)

# ---------------------------------------------------------------------------
# tests + benchmark
# ---------------------------------------------------------------------------
if get_option('tests') and not is_sub
  fm_test = executable('fm_test', 'tests/fm_test.c',
    c_args : common_args, dependencies : fatmap_dep)
  test('fm_test', fm_test, args : [meson.current_build_dir()], timeout : 300)

  # the SPIR-V test programs compiled ahead of time (fm3d_spirv_to_c at
  # build time; needs to run a host binary)
  aot_src  = []
  aot_args = []
  if get_option('shaders') and get_option('spirv') and meson.can_run_host_binaries()
    spirv_aot_gen = executable('spirv_aot_gen', 'tests/spirv_aot_gen.c',
      c_args : common_args, dependencies : fatmap_dep)
    aot_src  = custom_target('spirv_aot', output : 'spirv_aot.c', command : [spirv_aot_gen, '@OUTPUT@'])
    aot_args = ['-DFM_TEST_AOT=1']
  endif

  fm3d_test = executable('fm3d_test', 'tests/fm3d_test.c', aot_src,
    c_args : common_args + aot_args, dependencies : fatmap_dep)
  test('fm3d_test', fm3d_test, args : [meson.current_build_dir()], timeout : 300)

  # kernel tables against the scalar reference (internal symbols: static only)
  if get_option('default_library') == 'static'
    fm_kernel_test = executable('fm_kernel_test', 'tests/fm_kernel_test.c',
      c_args : common_args + simd_defs, include_directories : include_directories('src/core'),
      dependencies : fatmap_dep)
    test('fm_kernel_test', fm_kernel_test)
  endif

  executable('fm_bench', 'tests/fm_bench.c', aot_src,
    c_args : common_args + aot_args, dependencies : fatmap_dep)

  cpp_test_deps = [fatmapxx_dep]
  cpp_test_args = []
  if glm_dep.found()
    cpp_test_deps += glm_dep
    cpp_test_args += ['-DFM_TEST_GLM=1']
  endif
  fm_cpp_test = executable('fm_cpp_test', 'tests/cpp_test.cpp',
    cpp_args : cpp_test_args, dependencies : cpp_test_deps)
  test('fm_cpp_test', fm_cpp_test)
endif

# ---------------------------------------------------------------------------
# sandbox (SDL3 from external/sdl3.wrap, built with Meson)
# ---------------------------------------------------------------------------
sdl3_dep = dependency('sdl3', required : get_option('sandbox').disable_if(is_sub),
  default_options : ['default_library=static', 'warning_level=0'])
if sdl3_dep.found()
  # stb_image (single header in sandbox/stb) decodes the sample assets' JPEG / PNG
  stb_lib = static_library('stb_image', 'sandbox/stb/stb_image_impl.c',
    override_options : ['warning_level=0'])
  stb_dep = declare_dependency(link_with : stb_lib,
    include_directories : include_directories('sandbox/stb', is_system : true))
  # the sandbox's SPIR-V programs compiled ahead of time (G in the shader scenes)
  sb_aot_src  = []
  sb_aot_args = []
  if get_option('shaders') and get_option('spirv') and meson.can_run_host_binaries()
    sb_aot_gen  = executable('sandbox_aot_gen', 'sandbox/spirv_aot_gen.c',
      c_args : common_args, dependencies : fatmap_dep)
    sb_aot_src  = custom_target('sandbox_aot', output : 'sandbox_aot.c', command : [sb_aot_gen, '@OUTPUT@'])
    sb_aot_args = ['-DFM_SANDBOX_AOT=1']
  endif
  executable('fatmap_sandbox', 'sandbox/main.c', 'sandbox/obj_loader.c', 'sandbox/gltf_loader.c',
    'sandbox/image_load.c', sb_aot_src,
    c_args : common_args + sb_aot_args,
    dependencies : [fatmap_dep, sdl3_dep, stb_dep],
    win_subsystem : 'console')
endif

summary({
  'cpu' : cpu,
  'simd backends' : simd_defs,
  'threads' : threads_on,
  'features' : ['vbo=' + get_option('vbo').to_string(), 'tnl=' + get_option('tnl').to_string(),
                'shaders=' + get_option('shaders').to_string(),
                'spirv=' + (get_option('shaders') and get_option('spirv')).to_string()],
  'profiler' : get_option('profile'),
  'tracy' : get_option('tracy'),
  'glm (external)' : glm_dep.found(),
  'sandbox (SDL3)' : sdl3_dep.found(),
}, section : 'fatmap')

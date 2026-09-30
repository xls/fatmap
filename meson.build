project('fatmap', ['c', 'cpp'],
  version : '0.1.0',
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
pub_args = get_option('default_library') == 'static' ? ['-DFM_STATIC'] : []

m_dep = cc.find_library('m', required : false)

# threads are optional (-Dthreads=disabled for platforms without a thread
# library): command lists then execute serially on the calling thread
thread_dep = dependency('threads', required : get_option('threads'))
threads_on = thread_dep.found() and not get_option('threads').disabled()
thread_args = threads_on ? [] : ['-DFM_NO_THREADS']
thread_deps = threads_on ? [thread_dep] : []

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
  simd_libs += static_library('fm_avx2', 'src/core/fm_kernels_avx2.c',
    c_args : common_args + avx2_args + pub_args, include_directories : inc, pic : true)
  simd_defs += ['-DFM_HAVE_SSE2', '-DFM_HAVE_AVX2']
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
)
twod_src = files(
  'src/2d/fm2d_path.c',
  'src/2d/fm2d_stroke.c',
  'src/2d/fm2d_ctx.c',
)

fatmap_lib = library('fatmap', core_src + twod_src,
  c_args : common_args + simd_defs + profile_args + pub_args + thread_args + ['-DFM_BUILD'],
  include_directories : inc,
  link_whole : simd_libs,
  dependencies : [m_dep] + thread_deps + tracy_deps,
  gnu_symbol_visibility : 'hidden',
  install : true)

fatmap_dep = declare_dependency(
  link_with : fatmap_lib,
  include_directories : inc,
  compile_args : pub_args + profile_args,
  dependencies : [m_dep] + thread_deps + tracy_deps)
meson.override_dependency('fatmap', fatmap_dep)

install_subdir('include/fatmap', install_dir : get_option('includedir'))

# ---------------------------------------------------------------------------
# fatmap++ (C++17 header library wrapping 2D, later 3D)
# ---------------------------------------------------------------------------
fatmapxx_dep = declare_dependency(
  include_directories : include_directories('cpp/include'),
  dependencies : fatmap_dep)
meson.override_dependency('fatmapxx', fatmapxx_dep)
install_subdir('cpp/include/fatmap', install_dir : get_option('includedir'))

# glm (external/glm.wrap -> git clone, header only). Used by the C++ tests
# for layout/value compatibility with fm_math.h; the C library never needs it.
glm_dep = dependency('glm', required : false)

# ---------------------------------------------------------------------------
# tests + benchmark
# ---------------------------------------------------------------------------
if get_option('tests')
  fm_test = executable('fm_test', 'tests/fm_test.c',
    c_args : common_args, dependencies : fatmap_dep)
  test('fm_test', fm_test, args : [meson.current_build_dir()], timeout : 300)

  executable('fm_bench', 'tests/fm_bench.c',
    c_args : common_args, dependencies : fatmap_dep)

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
sdl3_dep = dependency('sdl3', required : get_option('sandbox'),
  default_options : ['default_library=static', 'warning_level=0'])
if sdl3_dep.found()
  executable('fatmap_sandbox', 'sandbox/main.c',
    c_args : common_args,
    dependencies : [fatmap_dep, sdl3_dep],
    win_subsystem : 'console')
endif

summary({
  'cpu' : cpu,
  'simd backends' : simd_defs,
  'threads' : threads_on,
  'profiler' : get_option('profile'),
  'tracy' : get_option('tracy'),
  'glm (external)' : glm_dep.found(),
  'sandbox (SDL3)' : sdl3_dep.found(),
}, section : 'fatmap')

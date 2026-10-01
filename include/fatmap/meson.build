# generated public header: the optional features of this build
fm_config_h = configure_file(input : 'fm_config.h.in', output : 'fm_config.h',
  configuration : {
    'FM_FEATURE_VBO' : get_option('vbo') ? 1 : 0,
    'FM_FEATURE_TNL' : get_option('tnl') ? 1 : 0,
    'FM_FEATURE_SHADERS' : get_option('shaders') ? 1 : 0,
    'FM_FEATURE_SPIRV' : get_option('shaders') and get_option('spirv') ? 1 : 0,
  },
  install : not meson.is_subproject(),
  install_dir : get_option('includedir') / 'fatmap')

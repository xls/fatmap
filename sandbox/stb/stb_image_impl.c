/* stb_image implementation for the sandbox (JPEG + PNG from memory only).
 * stb_image.h: https://github.com/nothings/stb at 2c980bb, public domain /
 * MIT (license at the end of the header). */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "stb_image.h"

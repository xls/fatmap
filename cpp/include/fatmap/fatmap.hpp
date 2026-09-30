// fatmap++ - C++17 wrapper around the fatmap C API.
//
//   fm::Surface  fb(1280, 720);
//   fm::Canvas2D ctx(fb);
//   ctx.fillStyle(fm::rgb(255, 0, 0));
//   ctx.beginPath(); ctx.arc(100, 100, 50, 0, 2 * fm::pi); ctx.fill();
//
// Canvas2D follows the HTML CanvasRenderingContext2D API; Canvas3D wraps the
// fixed function 3D pipeline (fm3d). Define FM_WITH_GLM (or include glm
// first) to pass glm matrices directly (zero-copy, same memory layout).
#pragma once

#include <fatmap/fatmap.h>

#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(FM_WITH_GLM) || defined(GLM_VERSION)
#  include <glm/glm.hpp>
#  define FM_HAS_GLM 1
#endif

namespace fm {

constexpr float pi = 3.14159265358979323846f;

using Color     = fm_color;
using BlendOp   = fm_blend_op;
using FillRule  = fm_fill_rule;
using AAMode    = fm_aa_mode;
using Matrix    = fm_affine;
using SimdLevel = fm_simd_level;

constexpr Color rgb(uint8_t r, uint8_t g, uint8_t b) { return FM_RGB(r, g, b); }
constexpr Color rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) { return FM_RGBA(r, g, b, a); }
inline Color    color(const char* css)
{
    fm_color c = 0;
    if (!fm_color_parse(css, &c)) throw std::invalid_argument(std::string("fatmap: bad color ") + css);
    return c;
}

// ---------------------------------------------------------------------------
class Surface {
public:
    Surface() = default;
    Surface(int w, int h, fm_format f = FM_FORMAT_ARGB32) : s_(fm_surface_create(w, h, f))
    {
        if (!s_) throw std::bad_alloc();
    }
    static Surface wrap(void* pixels, int w, int h, int stride_bytes, fm_format f = FM_FORMAT_ARGB32)
    {
        return Surface(fm_surface_wrap(pixels, w, h, stride_bytes, f));
    }
    static Surface fromRGBA8(const void* rgba, int w, int h, int stride_bytes)
    {
        return Surface(fm_surface_from_rgba8(rgba, w, h, stride_bytes));
    }
    Surface(const Surface&)            = delete;
    Surface& operator=(const Surface&) = delete;
    Surface(Surface&& o) noexcept : s_(std::exchange(o.s_, nullptr)) {}
    Surface& operator=(Surface&& o) noexcept
    {
        if (this != &o) {
            fm_surface_destroy(s_);
            s_ = std::exchange(o.s_, nullptr);
        }
        return *this;
    }
    ~Surface() { fm_surface_destroy(s_); }

    Surface clone() const { return Surface(fm_surface_clone(s_)); }
    Surface sub(int x, int y, int w, int h) const { return Surface(fm_surface_sub(s_, x, y, w, h)); }

    int       width() const { return s_->width; }
    int       height() const { return s_->height; }
    int       stride() const { return s_->stride; }
    uint32_t* row(int y) const { return fm_surface_row32(s_, y); }
    void*     data() const { return s_->data; }
    void      clear(Color c) { fm_surface_clear(s_, c); }
    Color     pixel(int x, int y) const { return fm_surface_get_pixel(s_, x, y); }
    bool      savePNG(const std::string& path) const { return fm_surface_write_png(s_, path.c_str()) != 0; }

    fm_surface*       get() { return s_; }
    const fm_surface* get() const { return s_; }
    explicit          operator bool() const { return s_ != nullptr; }

private:
    explicit Surface(fm_surface* s) : s_(s) {}
    fm_surface* s_ = nullptr;
};

// ---------------------------------------------------------------------------
// CanvasGradient / CanvasPattern (ref counted handle)
class Paint {
public:
    Paint() = default;
    explicit Paint(fm2d_paint* adopt) : p_(adopt) {}
    Paint(const Paint& o) : p_(fm2d_paint_retain(o.p_)) {}
    Paint(Paint&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    Paint& operator=(Paint o) noexcept
    {
        std::swap(p_, o.p_);
        return *this;
    }
    ~Paint() { fm2d_paint_release(p_); }

    Paint& addColorStop(float offset, Color c)
    {
        fm2d_paint_add_stop(p_, offset, c);
        return *this;
    }
    Paint& addColorStop(float offset, const char* css) { return addColorStop(offset, color(css)); }
    Paint& setTransform(const Matrix& m)
    {
        fm2d_paint_set_transform(p_, &m);
        return *this;
    }
    Paint& setExtend(fm_extend e)
    {
        fm2d_paint_set_extend(p_, e);
        return *this;
    }
    fm2d_paint* get() const { return p_; }

private:
    fm2d_paint* p_ = nullptr;
};

// ---------------------------------------------------------------------------
class Path2D {
public:
    Path2D() : p_(fm2d_path_create()) {}
    Path2D(const Path2D& o) : p_(fm2d_path_clone(o.p_)) {}
    Path2D(Path2D&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    Path2D& operator=(Path2D o) noexcept
    {
        std::swap(p_, o.p_);
        return *this;
    }
    ~Path2D() { fm2d_path_destroy(p_); }

    Path2D& addPath(const Path2D& o, const Matrix* m = nullptr) { fm2d_path_add_path(p_, o.p_, m); return *this; }
    Path2D& closePath() { fm2d_path_close(p_); return *this; }
    Path2D& moveTo(float x, float y) { fm2d_path_move_to(p_, x, y); return *this; }
    Path2D& lineTo(float x, float y) { fm2d_path_line_to(p_, x, y); return *this; }
    Path2D& quadraticCurveTo(float cx, float cy, float x, float y) { fm2d_path_quad_to(p_, cx, cy, x, y); return *this; }
    Path2D& bezierCurveTo(float c1x, float c1y, float c2x, float c2y, float x, float y)
    {
        fm2d_path_bezier_to(p_, c1x, c1y, c2x, c2y, x, y);
        return *this;
    }
    Path2D& arc(float x, float y, float r, float a0, float a1, bool ccw = false)
    {
        fm2d_path_arc(p_, x, y, r, a0, a1, ccw);
        return *this;
    }
    Path2D& arcTo(float x1, float y1, float x2, float y2, float r) { fm2d_path_arc_to(p_, x1, y1, x2, y2, r); return *this; }
    Path2D& ellipse(float x, float y, float rx, float ry, float rot, float a0, float a1, bool ccw = false)
    {
        fm2d_path_ellipse(p_, x, y, rx, ry, rot, a0, a1, ccw);
        return *this;
    }
    Path2D& rect(float x, float y, float w, float h) { fm2d_path_rect(p_, x, y, w, h); return *this; }
    Path2D& roundRect(float x, float y, float w, float h, std::initializer_list<float> radii)
    {
        std::vector<float> r(radii);
        fm2d_path_round_rect(p_, x, y, w, h, r.data(), (int)r.size());
        return *this;
    }
    const fm2d_path* get() const { return p_; }

private:
    fm2d_path* p_;
};

// ---------------------------------------------------------------------------
class Canvas2D {
public:
    explicit Canvas2D(Surface& target) : c_(fm2d_create(target.get()))
    {
        if (!c_) throw std::bad_alloc();
    }
    Canvas2D(const Canvas2D&)            = delete;
    Canvas2D& operator=(const Canvas2D&) = delete;
    Canvas2D(Canvas2D&& o) noexcept : c_(std::exchange(o.c_, nullptr)) {}
    ~Canvas2D() { fm2d_destroy(c_); }

    void setTarget(Surface& s) { fm2d_set_target(c_, s.get()); }
    void antialias(AAMode aa) { fm2d_set_antialias(c_, aa); }

    // state
    void save() { fm2d_save(c_); }
    void restore() { fm2d_restore(c_); }
    void reset() { fm2d_reset(c_); }

    // transforms
    void   translate(float x, float y) { fm2d_translate(c_, x, y); }
    void   scale(float x, float y) { fm2d_scale(c_, x, y); }
    void   rotate(float a) { fm2d_rotate(c_, a); }
    void   transform(float a, float b, float c, float d, float e, float f) { fm2d_transform(c_, a, b, c, d, e, f); }
    void   setTransform(float a, float b, float c, float d, float e, float f) { fm2d_set_transform(c_, a, b, c, d, e, f); }
    void   setTransform(const Matrix& m) { fm2d_set_transform(c_, m.a, m.b, m.c, m.d, m.e, m.f); }
    void   resetTransform() { fm2d_reset_transform(c_); }
    Matrix getTransform() const { return fm2d_get_transform(c_); }

    // compositing
    void  globalAlpha(float a) { fm2d_set_global_alpha(c_, a); }
    float globalAlpha() const { return fm2d_get_global_alpha(c_); }
    void  globalCompositeOperation(BlendOp op) { fm2d_set_composite_op(c_, op); }
    void  globalCompositeOperation(const char* name)
    {
        int op = fm_blend_op_from_name(name);
        if (op >= 0) fm2d_set_composite_op(c_, (fm_blend_op)op);
    }
    void imageSmoothingEnabled(bool on) { fm2d_set_image_smoothing(c_, on); }

    // styles
    void fillStyle(Color c) { fm2d_set_fill_color(c_, c); }
    void fillStyle(const char* css) { fm2d_set_fill_color(c_, color(css)); }
    void fillStyle(const Paint& p) { fm2d_set_fill_paint(c_, p.get()); }
    void strokeStyle(Color c) { fm2d_set_stroke_color(c_, c); }
    void strokeStyle(const char* css) { fm2d_set_stroke_color(c_, color(css)); }
    void strokeStyle(const Paint& p) { fm2d_set_stroke_paint(c_, p.get()); }
    void lineWidth(float w) { fm2d_set_line_width(c_, w); }
    void lineCap(fm2d_line_cap c) { fm2d_set_line_cap(c_, c); }
    void lineJoin(fm2d_line_join j) { fm2d_set_line_join(c_, j); }
    void miterLimit(float l) { fm2d_set_miter_limit(c_, l); }
    void setLineDash(std::initializer_list<float> d)
    {
        std::vector<float> v(d);
        fm2d_set_line_dash(c_, v.data(), (int)v.size());
    }
    void setLineDash(const std::vector<float>& d) { fm2d_set_line_dash(c_, d.data(), (int)d.size()); }
    void lineDashOffset(float o) { fm2d_set_line_dash_offset(c_, o); }

    // paints
    static Paint createLinearGradient(float x0, float y0, float x1, float y1)
    {
        return Paint(fm2d_paint_linear(x0, y0, x1, y1));
    }
    static Paint createRadialGradient(float x0, float y0, float r0, float x1, float y1, float r1)
    {
        return Paint(fm2d_paint_radial(x0, y0, r0, x1, y1, r1));
    }
    static Paint createConicGradient(float startAngle, float x, float y)
    {
        return Paint(fm2d_paint_conic(startAngle, x, y));
    }
    static Paint createPattern(const Surface& img, fm2d_repetition rep = FM2D_REPEAT)
    {
        return Paint(fm2d_paint_pattern(img.get(), rep));
    }

    // path
    void beginPath() { fm2d_begin_path(c_); }
    void closePath() { fm2d_close_path(c_); }
    void moveTo(float x, float y) { fm2d_move_to(c_, x, y); }
    void lineTo(float x, float y) { fm2d_line_to(c_, x, y); }
    void quadraticCurveTo(float cx, float cy, float x, float y) { fm2d_quad_to(c_, cx, cy, x, y); }
    void bezierCurveTo(float c1x, float c1y, float c2x, float c2y, float x, float y)
    {
        fm2d_bezier_to(c_, c1x, c1y, c2x, c2y, x, y);
    }
    void arc(float x, float y, float r, float a0, float a1, bool ccw = false) { fm2d_arc(c_, x, y, r, a0, a1, ccw); }
    void arcTo(float x1, float y1, float x2, float y2, float r) { fm2d_arc_to(c_, x1, y1, x2, y2, r); }
    void ellipse(float x, float y, float rx, float ry, float rot, float a0, float a1, bool ccw = false)
    {
        fm2d_ellipse(c_, x, y, rx, ry, rot, a0, a1, ccw);
    }
    void rect(float x, float y, float w, float h) { fm2d_rect(c_, x, y, w, h); }
    void roundRect(float x, float y, float w, float h, std::initializer_list<float> radii)
    {
        std::vector<float> r(radii);
        fm2d_round_rect(c_, x, y, w, h, r.data(), (int)r.size());
    }

    // drawing
    void fill(FillRule r = FM_FILL_NONZERO) { fm2d_fill(c_, r); }
    void fill(const Path2D& p, FillRule r = FM_FILL_NONZERO) { fm2d_fill_path(c_, p.get(), r); }
    void stroke() { fm2d_stroke(c_); }
    void stroke(const Path2D& p) { fm2d_stroke_path(c_, p.get()); }
    void clip(FillRule r = FM_FILL_NONZERO) { fm2d_clip(c_, r); }
    void clip(const Path2D& p, FillRule r = FM_FILL_NONZERO) { fm2d_clip_path(c_, p.get(), r); }
    void fillRect(float x, float y, float w, float h) { fm2d_fill_rect(c_, x, y, w, h); }
    void strokeRect(float x, float y, float w, float h) { fm2d_stroke_rect(c_, x, y, w, h); }
    void clearRect(float x, float y, float w, float h) { fm2d_clear_rect(c_, x, y, w, h); }
    bool isPointInPath(float x, float y, FillRule r = FM_FILL_NONZERO) { return fm2d_is_point_in_path(c_, x, y, r) != 0; }
    bool isPointInStroke(float x, float y) { return fm2d_is_point_in_stroke(c_, x, y) != 0; }
    void drawImage(const Surface& img, float dx, float dy) { fm2d_draw_image(c_, img.get(), dx, dy); }
    void drawImage(const Surface& img, float dx, float dy, float dw, float dh)
    {
        fm2d_draw_image_scaled(c_, img.get(), dx, dy, dw, dh);
    }
    void drawImage(const Surface& img, float sx, float sy, float sw, float sh, float dx, float dy, float dw, float dh)
    {
        fm2d_draw_image_sub(c_, img.get(), sx, sy, sw, sh, dx, dy, dw, dh);
    }

    fm2d_ctx* get() { return c_; }

private:
    fm2d_ctx* c_;
};

// ---------------------------------------------------------------------------
// SIMD control + profiler
inline SimdLevel   simdBest() { return fm_simd_best(); }
inline bool        setSimd(SimdLevel l) { return fm_simd_set(l) != 0; }
inline const char* simdName(SimdLevel l) { return fm_simd_name(l); }
inline std::string profilerReport()
{
    std::string s(8192, '\0');
    int         n = fm_prof_report(&s[0], (int)s.size());
    s.resize((size_t)(n > 0 ? n : 0));
    return s;
}

#ifdef FM_HAS_GLM
// fm_math types share glm's memory layout (checked at compile time).
static_assert(sizeof(fm_mat4) == sizeof(glm::mat4), "fm_mat4 / glm::mat4 layout mismatch");
static_assert(sizeof(fm_vec4) == sizeof(glm::vec4), "fm_vec4 / glm::vec4 layout mismatch");
static_assert(sizeof(fm_vec3) == sizeof(glm::vec3), "fm_vec3 / glm::vec3 layout mismatch");
inline fm_mat4 toFm(const glm::mat4& m)
{
    fm_mat4 r;
    std::memcpy(static_cast<void*>(&r), static_cast<const void*>(&m), sizeof(r));
    return r;
}
inline glm::mat4 toGlm(const fm_mat4& m)
{
    glm::mat4 r;
    std::memcpy(static_cast<void*>(&r), static_cast<const void*>(&m), sizeof(r));
    return r;
}
inline fm_vec3   toFm(const glm::vec3& v) { return fm_v3(v.x, v.y, v.z); }
inline glm::vec3 toGlm(const fm_vec3& v) { return glm::vec3(v.x, v.y, v.z); }
#endif

// ---------------------------------------------------------------------------
// Executor (thread pool) and swapchain
class Executor {
public:
    explicit Executor(int threads = 0) : e_(fm_executor_create(threads)) {}
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;
    ~Executor() { fm_executor_destroy(e_); }
    int          workers() const { return e_ ? e_->workers : 1; }
    fm_executor* get() const { return e_; }

private:
    fm_executor* e_;
};

class Swapchain {
public:
    Swapchain(int w, int h, int buffers = 2, bool depth = true) : s_(fm_swapchain_create(w, h, buffers, depth))
    {
        if (!s_) throw std::bad_alloc();
    }
    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    ~Swapchain() { fm_swapchain_destroy(s_); }
    fm_surface* back() const { return fm_swapchain_back(s_); }
    fm_surface* depth() const { return fm_swapchain_depth(s_); }
    fm_surface* front() const { return fm_swapchain_front(s_); }
    fm_surface* present() { return fm_swapchain_present(s_); }

private:
    fm_swapchain* s_;
};

// ---------------------------------------------------------------------------
// 3D
using Vertex3D = fm3d_vertex;
using Sampler  = fm3d_sampler;

class Texture {
public:
    Texture() = default;
    explicit Texture(const Surface& img, bool mipmaps = true) : t_(fm3d_texture_create(img.get(), mipmaps))
    {
        if (!t_) throw std::runtime_error("fatmap: texture creation failed");
    }
    Texture(const Texture& o) : t_(fm3d_texture_retain(o.t_)) {}
    Texture(Texture&& o) noexcept : t_(std::exchange(o.t_, nullptr)) {}
    Texture& operator=(Texture o) noexcept
    {
        std::swap(t_, o.t_);
        return *this;
    }
    ~Texture() { fm3d_texture_release(t_); }
    int           levels() const { return fm3d_texture_levels(t_); }
    fm3d_texture* get() const { return t_; }

private:
    fm3d_texture* t_ = nullptr;
};

class Canvas3D {
public:
    Canvas3D() : c_(fm3d_create())
    {
        if (!c_) throw std::bad_alloc();
    }
    Canvas3D(const Canvas3D&)            = delete;
    Canvas3D& operator=(const Canvas3D&) = delete;
    ~Canvas3D() { fm3d_destroy(c_); }

    void setTarget(fm_surface* color, fm_surface* depth = nullptr) { fm3d_set_target(c_, color, depth); }
    void setTarget(Surface& color, Surface* depth = nullptr) { fm3d_set_target(c_, color.get(), depth ? depth->get() : nullptr); }
    void setTarget(Swapchain& sc) { fm3d_set_target(c_, sc.back(), sc.depth()); }

    void setModel(const fm_mat4& m) { fm3d_set_model(c_, &m); }
    void setView(const fm_mat4& m) { fm3d_set_view(c_, &m); }
    void setProjection(const fm_mat4& m) { fm3d_set_projection(c_, &m); }
#ifdef FM_HAS_GLM
    void setModel(const glm::mat4& m) { setModel(toFm(m)); }
    void setView(const glm::mat4& m) { setView(toFm(m)); }
    void setProjection(const glm::mat4& m) { setProjection(toFm(m)); }
#endif
    void clipDepth(fm3d_clip_depth m) { fm3d_set_clip_depth(c_, m); }
    void viewport(int x, int y, int w, int h) { fm3d_set_viewport(c_, x, y, w, h); }
    void scissor(int x, int y, int w, int h) { fm3d_set_scissor(c_, 1, x, y, w, h); }
    void noScissor() { fm3d_set_scissor(c_, 0, 0, 0, 0, 0); }
    void cull(fm3d_cull c, fm3d_winding front = FM3D_FRONT_CCW) { fm3d_set_cull(c_, c, front); }
    void perspectiveCorrect(bool on) { fm3d_set_perspective_correct(c_, on); }
    void depthTest(fm3d_compare f, bool write = true) { fm3d_set_depth_test(c_, f, write); }
    void depthBias(float factor, float units) { fm3d_set_depth_bias(c_, factor, units); }
    void depthRange(float n, float f) { fm3d_set_depth_range(c_, n, f); }
    void depthClamp(bool on) { fm3d_set_depth_clamp(c_, on); }
    void stencilBuffer(Surface& s) { fm3d_set_stencil_buffer(c_, s.get()); }
    void stencilTest(bool on) { fm3d_set_stencil_test(c_, on); }
    void stencilFunc(fm3d_compare f, uint8_t ref, uint8_t mask = 0xff, fm3d_face face = FM3D_FACE_FRONT_AND_BACK)
    {
        fm3d_set_stencil_func(c_, face, f, ref, mask);
    }
    void stencilOp(fm3d_stencil_op sfail, fm3d_stencil_op dpfail, fm3d_stencil_op dppass,
                   fm3d_face face = FM3D_FACE_FRONT_AND_BACK)
    {
        fm3d_set_stencil_op(c_, face, sfail, dpfail, dppass);
    }
    void stencilWriteMask(uint8_t m, fm3d_face face = FM3D_FACE_FRONT_AND_BACK) { fm3d_set_stencil_write_mask(c_, face, m); }
    void colorWrite(bool on) { fm3d_set_color_write(c_, on); }
    void clearStencil(uint8_t v = 0) { fm3d_clear_stencil(c_, v); }
    void texture(const Texture& t, const Sampler& s) { fm3d_set_texture(c_, t.get(), &s); }
    void noTexture() { fm3d_set_texture(c_, nullptr, nullptr); }
    void texenv(fm3d_texenv e) { fm3d_set_texenv(c_, e); }
    void alphaTest(fm3d_compare f, float ref) { fm3d_set_alpha_test(c_, f, ref); }
    void blend(BlendOp op) { fm3d_set_blend(c_, op); }
    void opacity(float a) { fm3d_set_opacity(c_, a); }

    void clearColor(Color c) { fm3d_clear_color(c_, c); }
    void clearDepth(float d = 1.0f) { fm3d_clear_depth(c_, d); }
    void draw(const Vertex3D* v, int n) { fm3d_draw(c_, v, n); }
    void draw(const std::vector<Vertex3D>& v) { fm3d_draw(c_, v.data(), (int)v.size()); }
    void drawIndexed(const std::vector<Vertex3D>& v, const std::vector<uint32_t>& idx)
    {
        fm3d_draw_indexed(c_, v.data(), (int)v.size(), idx.data(), (int)idx.size());
    }

    void deferred(bool on) { fm3d_set_deferred(c_, on); }
    void executor(Executor& e) { fm3d_set_executor(c_, e.get()); }
    void tileSize(int px) { fm3d_set_tile_size(c_, px); }
    void flush() { fm3d_flush(c_); }
    fm3d_stats stats() { return fm3d_get_stats(c_); }

    fm3d_ctx* get() { return c_; }

private:
    fm3d_ctx* c_;
};

} // namespace fm

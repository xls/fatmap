/*
 * fatmap - SIMD kernel template.
 *
 * Included once per backend (scalar, SSE2, AVX2, NEON). The backend defines
 * the vector primitives below, this file defines every op on top of them so
 * the math (and therefore the output) is identical on every backend.
 *
 * Backend contract:
 *   FMK(name)          name mangling, e.g. name##_sse2
 *   FMK_TABLE          name of the fm_kernels table to emit
 *   FMK_LEVEL          fm_simd_level
 *   FMK_PX             pixels per block
 *   vpx                a block of FMK_PX ARGB32 pixels
 *   vw                 half a block widened to u16 lanes (B,G,R,A per pixel)
 *   vpx_load/store/set1, vw_lo/hi/pack, vpx_mask_load (coverage byte
 *   replicated into all 4 channels), vw_add/sub/mul/div255/min/max/set1/shr8,
 *   vw_alpha (broadcast alpha lane), vw_alpha_merge(c, a) (alpha lane from a),
 *   fmk_mask_zero/fmk_mask_full (FMK_PX coverage bytes), vpx_all_opaque,
 *   vpx_all_zero.
 *   FMK(accumulate), FMK(bilinear), FMK(linear_grad) defined by the backend.
 *
 * Values fed to vw_pack are always <= 32767 so signed (SSE) and unsigned
 * (NEON) saturation agree.
 */

FM_INLINE vw fmk_inv(vw x) { return vw_sub(vw_set1(255), x); }
FM_INLINE vw fmk_md(vw a, vw b) { return vw_div255(vw_mul(a, b)); }
FM_INLINE vw fmk_over(vw s, vw d) { return vw_add(s, fmk_md(d, fmk_inv(vw_alpha(s)))); }
/* lerp(d, r, m) */
FM_INLINE vw fmk_lerp(vw d, vw r, vw m) { return vw_div255(vw_add(vw_mul(r, m), vw_mul(d, fmk_inv(m)))); }

#define FMK_TAIL_BEGIN(n, i)          \
    {                                 \
        int      r_ = (n) - (i);      \
        uint32_t td_[FMK_PX];         \
        uint32_t ts_[FMK_PX];         \
        uint8_t  tm_[FMK_PX];         \
        memset(td_, 0, sizeof(td_));  \
        memset(ts_, 0, sizeof(ts_));  \
        memset(tm_, 0, sizeof(tm_));  \
        (void)ts_;                    \
        (void)tm_;
#define FMK_TAIL_END }

/* ---- fill ----------------------------------------------------------------- */

static void FMK(fill)(uint32_t* d, uint32_t v, int n)
{
    vpx vv = vpx_set1(v);
    int i  = 0;
    for (; i + FMK_PX <= n; i += FMK_PX) vpx_store(d + i, vv);
    for (; i < n; i++) d[i] = v;
}

/* ---- solid source over ------------------------------------------------------ */

FM_INLINE void fmk_blk_solid_over(uint32_t* d, vw s0, vw s1, vw ia)
{
    vpx dp = vpx_load(d);
    vw  d0 = vw_add(s0, fmk_md(vw_lo(dp), ia));
    vw  d1 = vw_add(s1, fmk_md(vw_hi(dp), ia));
    vpx_store(d, vw_pack(d0, d1));
}

static void FMK(solid_over)(uint32_t* d, uint32_t s, int n)
{
    if ((s >> 24) == 255) {
        FMK(fill)(d, s, n);
        return;
    }
    if (s == 0) return;
    vpx sv = vpx_set1(s);
    vw  s0 = vw_lo(sv), s1 = vw_hi(sv);
    vw  ia = fmk_inv(vw_alpha(s0));
    int i  = 0;
    for (; i + FMK_PX <= n; i += FMK_PX) fmk_blk_solid_over(d + i, s0, s1, ia);
    if (i < n) {
        FMK_TAIL_BEGIN(n, i)
        memcpy(td_, d + i, (size_t)r_ * 4);
        fmk_blk_solid_over(td_, s0, s1, ia);
        memcpy(d + i, td_, (size_t)r_ * 4);
        FMK_TAIL_END
    }
}

FM_INLINE void fmk_blk_solid_over_mask(uint32_t* d, vpx sv, vw s0, vw s1, int opaque, const uint8_t* m)
{
    if (fmk_mask_zero(m)) return;
    if (opaque && fmk_mask_full(m)) {
        vpx_store(d, sv);
        return;
    }
    vpx mp = vpx_mask_load(m);
    vpx dp = vpx_load(d);
    vw  a0 = fmk_md(s0, vw_lo(mp));
    vw  a1 = fmk_md(s1, vw_hi(mp));
    vpx_store(d, vw_pack(fmk_over(a0, vw_lo(dp)), fmk_over(a1, vw_hi(dp))));
}

static void FMK(solid_over_mask)(uint32_t* d, uint32_t s, const uint8_t* m, int n)
{
    if (s == 0) return;
    vpx sv     = vpx_set1(s);
    vw  s0     = vw_lo(sv), s1 = vw_hi(sv);
    int opaque = (s >> 24) == 255;
    int i      = 0;
    for (; i + FMK_PX <= n; i += FMK_PX) fmk_blk_solid_over_mask(d + i, sv, s0, s1, opaque, m + i);
    if (i < n) {
        FMK_TAIL_BEGIN(n, i)
        memcpy(td_, d + i, (size_t)r_ * 4);
        memcpy(tm_, m + i, (size_t)r_);
        fmk_blk_solid_over_mask(td_, sv, s0, s1, opaque, tm_);
        memcpy(d + i, td_, (size_t)r_ * 4);
        FMK_TAIL_END
    }
}

/* ---- span source over --------------------------------------------------------- */

FM_INLINE void fmk_blk_span_over(uint32_t* d, const uint32_t* s)
{
    vpx sp = vpx_load(s);
    if (vpx_all_opaque(sp)) {
        vpx_store(d, sp);
        return;
    }
    if (vpx_all_zero(sp)) return;
    vpx dp = vpx_load(d);
    vpx_store(d, vw_pack(fmk_over(vw_lo(sp), vw_lo(dp)), fmk_over(vw_hi(sp), vw_hi(dp))));
}

static void FMK(span_over)(uint32_t* d, const uint32_t* s, int n)
{
    int i = 0;
    for (; i + FMK_PX <= n; i += FMK_PX) fmk_blk_span_over(d + i, s + i);
    if (i < n) {
        FMK_TAIL_BEGIN(n, i)
        memcpy(td_, d + i, (size_t)r_ * 4);
        memcpy(ts_, s + i, (size_t)r_ * 4);
        fmk_blk_span_over(td_, ts_);
        memcpy(d + i, td_, (size_t)r_ * 4);
        FMK_TAIL_END
    }
}

FM_INLINE void fmk_blk_span_over_mask(uint32_t* d, const uint32_t* s, const uint8_t* m)
{
    if (fmk_mask_zero(m)) return;
    if (fmk_mask_full(m)) {
        fmk_blk_span_over(d, s);
        return;
    }
    vpx sp = vpx_load(s);
    vpx mp = vpx_mask_load(m);
    vpx dp = vpx_load(d);
    vw  a0 = fmk_md(vw_lo(sp), vw_lo(mp));
    vw  a1 = fmk_md(vw_hi(sp), vw_hi(mp));
    vpx_store(d, vw_pack(fmk_over(a0, vw_lo(dp)), fmk_over(a1, vw_hi(dp))));
}

static void FMK(span_over_mask)(uint32_t* d, const uint32_t* s, const uint8_t* m, int n)
{
    int i = 0;
    for (; i + FMK_PX <= n; i += FMK_PX) fmk_blk_span_over_mask(d + i, s + i, m + i);
    if (i < n) {
        FMK_TAIL_BEGIN(n, i)
        memcpy(td_, d + i, (size_t)r_ * 4);
        memcpy(ts_, s + i, (size_t)r_ * 4);
        memcpy(tm_, m + i, (size_t)r_);
        fmk_blk_span_over_mask(td_, ts_, tm_);
        memcpy(d + i, td_, (size_t)r_ * 4);
        FMK_TAIL_END
    }
}

/* ---- generic ops ---------------------------------------------------------------
 * Each op computes r = op(s, d) on u16 lanes (sa/da = broadcast alphas),
 * then dst = lerp(d, min(r,255), coverage). */

FM_INLINE vw fmk_op_src_in(vw s, vw d, vw sa, vw da) { (void)d; (void)sa; return fmk_md(s, da); }
FM_INLINE vw fmk_op_src_out(vw s, vw d, vw sa, vw da) { (void)d; (void)sa; return fmk_md(s, fmk_inv(da)); }
FM_INLINE vw fmk_op_src_atop(vw s, vw d, vw sa, vw da) { return vw_add(fmk_md(s, da), fmk_md(d, fmk_inv(sa))); }
FM_INLINE vw fmk_op_dst_over(vw s, vw d, vw sa, vw da) { (void)sa; return vw_add(d, fmk_md(s, fmk_inv(da))); }
FM_INLINE vw fmk_op_dst_in(vw s, vw d, vw sa, vw da) { (void)s; (void)da; return fmk_md(d, sa); }
FM_INLINE vw fmk_op_dst_out(vw s, vw d, vw sa, vw da) { (void)s; (void)da; return fmk_md(d, fmk_inv(sa)); }
FM_INLINE vw fmk_op_dst_atop(vw s, vw d, vw sa, vw da) { return vw_add(fmk_md(s, fmk_inv(da)), fmk_md(d, sa)); }
FM_INLINE vw fmk_op_lighter(vw s, vw d, vw sa, vw da) { (void)sa; (void)da; return vw_add(s, d); }
FM_INLINE vw fmk_op_copy(vw s, vw d, vw sa, vw da) { (void)d; (void)sa; (void)da; return s; }
FM_INLINE vw fmk_op_xor(vw s, vw d, vw sa, vw da) { return vw_add(fmk_md(s, fmk_inv(da)), fmk_md(d, fmk_inv(sa))); }
FM_INLINE vw fmk_op_clear(vw s, vw d, vw sa, vw da) { (void)s; (void)d; (void)sa; (void)da; return vw_set1(0); }
FM_INLINE vw fmk_op_multiply(vw s, vw d, vw sa, vw da)
{
    /* s(1-da) + d(1-sa) + s*d  ==  s(1-da) + d(1 - sa + s) */
    return vw_add(fmk_md(s, fmk_inv(da)), fmk_md(d, vw_add(fmk_inv(sa), s)));
}
FM_INLINE vw fmk_op_screen(vw s, vw d, vw sa, vw da) { (void)sa; (void)da; return vw_sub(vw_add(s, d), fmk_md(s, d)); }
FM_INLINE vw fmk_op_darken(vw s, vw d, vw sa, vw da)
{
    return vw_sub(vw_add(s, d), vw_div255(vw_max(vw_mul(s, da), vw_mul(d, sa))));
}
FM_INLINE vw fmk_op_lighten(vw s, vw d, vw sa, vw da)
{
    return vw_sub(vw_add(s, d), vw_div255(vw_min(vw_mul(s, da), vw_mul(d, sa))));
}
FM_INLINE vw fmk_op_difference(vw s, vw d, vw sa, vw da)
{
    vw sum = vw_add(s, d);
    vw mn  = vw_div255(vw_min(vw_mul(s, da), vw_mul(d, sa)));
    return vw_alpha_merge(vw_sub(sum, vw_add(mn, mn)), vw_sub(sum, fmk_md(s, d)));
}
FM_INLINE vw fmk_op_exclusion(vw s, vw d, vw sa, vw da)
{
    (void)sa;
    (void)da;
    vw sum = vw_add(s, d);
    vw sd  = fmk_md(s, d);
    return vw_alpha_merge(vw_sub(sum, vw_add(sd, sd)), vw_sub(sum, sd));
}

#define FMK_DEFINE_OP(NAME)                                                                          \
    FM_INLINE vw fmk_opw_##NAME(vw s, vw d, vw m, int has_m)                                         \
    {                                                                                                \
        vw r = vw_min(fmk_op_##NAME(s, d, vw_alpha(s), vw_alpha(d)), vw_set1(255));                 \
        return has_m ? fmk_lerp(d, r, m) : r;                                                        \
    }                                                                                                \
    FM_INLINE void fmk_blk_##NAME(uint32_t* d, const uint32_t* s, const uint8_t* m)                  \
    {                                                                                                \
        int has_m = 0;                                                                               \
        vpx mp    = vpx_set1(0xffffffffu);                                                           \
        if (m) {                                                                                     \
            if (fmk_mask_zero(m)) return;                                                            \
            if (!fmk_mask_full(m)) {                                                                 \
                has_m = 1;                                                                           \
                mp    = vpx_mask_load(m);                                                            \
            }                                                                                        \
        }                                                                                            \
        vpx sp = vpx_load(s);                                                                        \
        vpx dp = vpx_load(d);                                                                        \
        vw  r0 = fmk_opw_##NAME(vw_lo(sp), vw_lo(dp), vw_lo(mp), has_m);                             \
        vw  r1 = fmk_opw_##NAME(vw_hi(sp), vw_hi(dp), vw_hi(mp), has_m);                             \
        vpx_store(d, vw_pack(r0, r1));                                                               \
    }                                                                                                \
    static void FMK(span_##NAME)(uint32_t * d, const uint32_t* s, const uint8_t* m, int n)           \
    {                                                                                                \
        int i = 0;                                                                                   \
        for (; i + FMK_PX <= n; i += FMK_PX) fmk_blk_##NAME(d + i, s + i, m ? m + i : NULL);          \
        if (i < n) {                                                                                 \
            FMK_TAIL_BEGIN(n, i)                                                                     \
            memcpy(td_, d + i, (size_t)r_ * 4);                                                      \
            memcpy(ts_, s + i, (size_t)r_ * 4);                                                      \
            if (m) memcpy(tm_, m + i, (size_t)r_);                                                   \
            fmk_blk_##NAME(td_, ts_, m ? tm_ : NULL);                                                \
            memcpy(d + i, td_, (size_t)r_ * 4);                                                      \
            FMK_TAIL_END                                                                             \
        }                                                                                            \
    }

FMK_DEFINE_OP(src_in)
FMK_DEFINE_OP(src_out)
FMK_DEFINE_OP(src_atop)
FMK_DEFINE_OP(dst_over)
FMK_DEFINE_OP(dst_in)
FMK_DEFINE_OP(dst_out)
FMK_DEFINE_OP(dst_atop)
FMK_DEFINE_OP(lighter)
FMK_DEFINE_OP(copy)
FMK_DEFINE_OP(xor)
FMK_DEFINE_OP(clear)
FMK_DEFINE_OP(multiply)
FMK_DEFINE_OP(screen)
FMK_DEFINE_OP(darken)
FMK_DEFINE_OP(lighten)
FMK_DEFINE_OP(difference)
FMK_DEFINE_OP(exclusion)

static void FMK(span_op)(uint32_t* d, const uint32_t* s, const uint8_t* m, int n, int op)
{
    switch (op) {
    case FM_OP_SRC_OVER:
        if (m)
            FMK(span_over_mask)(d, s, m, n);
        else
            FMK(span_over)(d, s, n);
        break;
    case FM_OP_SRC_IN: FMK(span_src_in)(d, s, m, n); break;
    case FM_OP_SRC_OUT: FMK(span_src_out)(d, s, m, n); break;
    case FM_OP_SRC_ATOP: FMK(span_src_atop)(d, s, m, n); break;
    case FM_OP_DST_OVER: FMK(span_dst_over)(d, s, m, n); break;
    case FM_OP_DST_IN: FMK(span_dst_in)(d, s, m, n); break;
    case FM_OP_DST_OUT: FMK(span_dst_out)(d, s, m, n); break;
    case FM_OP_DST_ATOP: FMK(span_dst_atop)(d, s, m, n); break;
    case FM_OP_LIGHTER: FMK(span_lighter)(d, s, m, n); break;
    case FM_OP_COPY: FMK(span_copy)(d, s, m, n); break;
    case FM_OP_XOR: FMK(span_xor)(d, s, m, n); break;
    case FM_OP_CLEAR: FMK(span_clear)(d, s, m, n); break;
    case FM_OP_MULTIPLY: FMK(span_multiply)(d, s, m, n); break;
    case FM_OP_SCREEN: FMK(span_screen)(d, s, m, n); break;
    case FM_OP_DARKEN: FMK(span_darken)(d, s, m, n); break;
    case FM_OP_LIGHTEN: FMK(span_lighten)(d, s, m, n); break;
    case FM_OP_DIFFERENCE: FMK(span_difference)(d, s, m, n); break;
    case FM_OP_EXCLUSION: FMK(span_exclusion)(d, s, m, n); break;
    default: fm_span_op_complex(d, s, m, n, op); break;
    }
}

/* ---- coverage mask ops (bytes processed as 4-byte "pixels") ------------------------ */

static void FMK(mask_mul)(uint8_t* m, const uint8_t* c, int n)
{
    int i = 0;
    for (; i + FMK_PX * 4 <= n; i += FMK_PX * 4) {
        vpx a = vpx_load(m + i), b = vpx_load(c + i);
        vpx_store(m + i, vw_pack(fmk_md(vw_lo(a), vw_lo(b)), fmk_md(vw_hi(a), vw_hi(b))));
    }
    for (; i < n; i++) m[i] = (uint8_t)fm_div255((uint32_t)m[i] * c[i]);
}

static void FMK(mask_scale)(uint8_t* m, uint32_t k, int n)
{
    vw  kk = vw_set1((uint16_t)k);
    int i  = 0;
    for (; i + FMK_PX * 4 <= n; i += FMK_PX * 4) {
        vpx a = vpx_load(m + i);
        vpx_store(m + i, vw_pack(fmk_md(vw_lo(a), kk), fmk_md(vw_hi(a), kk)));
    }
    for (; i < n; i++) m[i] = (uint8_t)fm_div255((uint32_t)m[i] * k);
}

/* ---- 3D texenv combine + weighted lerp ------------------------------------------------ */

FM_INLINE vw fmk_combine16(int env, vw t, vw c)
{
    switch (env) {
    case 1: return t;                                                             /* replace */
    case 2: return vw_min(vw_add(t, fmk_md(c, fmk_inv(vw_alpha(t)))), vw_set1(255)); /* decal: t over c */
    case 3: {                                                                     /* add */
        vw s = vw_add(t, c);
        return vw_min(s, vw_alpha(vw_min(s, vw_set1(255))));
    }
    default: return fmk_md(t, c); /* modulate */
    }
}

FM_INLINE void fmk_blk_combine(int env, const uint32_t* t, const uint32_t* c, uint32_t* o)
{
    vpx tp = vpx_load(t), cp = vpx_load(c);
    vpx_store(o, vw_pack(fmk_combine16(env, vw_lo(tp), vw_lo(cp)), fmk_combine16(env, vw_hi(tp), vw_hi(cp))));
}

static void FMK(combine)(int env, const uint32_t* t, const uint32_t* c, int n, uint32_t* out)
{
    int i = 0;
    for (; i + FMK_PX <= n; i += FMK_PX) fmk_blk_combine(env, t + i, c + i, out + i);
    if (i < n) {
        FMK_TAIL_BEGIN(n, i)
        memcpy(td_, t + i, (size_t)r_ * 4);
        memcpy(ts_, c + i, (size_t)r_ * 4);
        fmk_blk_combine(env, td_, ts_, td_);
        memcpy(out + i, td_, (size_t)r_ * 4);
        FMK_TAIL_END
    }
}

FM_INLINE vw fmk_lerp8w(vw a, vw b, vw f) { return vw_shr8(vw_add(vw_mul(a, vw_sub(vw_set1(256), f)), vw_mul(b, f))); }

FM_INLINE void fmk_blk_lerp8(const uint32_t* a, const uint32_t* b, const uint8_t* f, uint32_t* o)
{
    vpx ap = vpx_load(a), bp = vpx_load(b), fp = vpx_mask_load(f);
    vpx_store(o, vw_pack(fmk_lerp8w(vw_lo(ap), vw_lo(bp), vw_lo(fp)), fmk_lerp8w(vw_hi(ap), vw_hi(bp), vw_hi(fp))));
}

static void FMK(lerp8)(const uint32_t* a, const uint32_t* b, const uint8_t* f, int n, uint32_t* out)
{
    int i = 0;
    for (; i + FMK_PX <= n; i += FMK_PX) fmk_blk_lerp8(a + i, b + i, f + i, out + i);
    if (i < n) {
        FMK_TAIL_BEGIN(n, i)
        memcpy(td_, a + i, (size_t)r_ * 4);
        memcpy(ts_, b + i, (size_t)r_ * 4);
        memcpy(tm_, f + i, (size_t)r_);
        fmk_blk_lerp8(td_, ts_, tm_, td_);
        memcpy(out + i, td_, (size_t)r_ * 4);
        FMK_TAIL_END
    }
}

/* ---- table ------------------------------------------------------------------------ */

const fm_kernels FMK_TABLE = {
    FMK_LEVEL,
    FMK(fill),
    FMK(solid_over),
    FMK(solid_over_mask),
    FMK(span_over),
    FMK(span_over_mask),
    FMK(span_op),
    FMK(mask_mul),
    FMK(mask_scale),
    FMK(accumulate),
    FMK(bilinear),
    FMK(bilinear_pts),
    FMK(depth_f32),
    FMK(skin4),
    FMK(depth_ms),
    FMK(resolve),
    FMK(texcoord),
    FMK(premul_f),
    FMK(combine),
    FMK(lerp8),
    FMK(linear_grad),
    FMK(acc_add),
    FMK(plane),
    FMK(plane_recip),
    FMK(plane_mul),
};

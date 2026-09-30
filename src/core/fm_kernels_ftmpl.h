/*
 * fatmap - float SIMD kernel template (vertex stage math).
 *
 * Included once per backend, before fm_kernels_tmpl.h. The backend defines
 * a float vector type and these primitives, so every backend (including the
 * 1 wide scalar reference) runs the same operations in the same order and
 * produces the same bits:
 *
 *   FMF_W            lanes per vector
 *   vf, vm           float vector, compare mask
 *   vf_set(f) vf_ld(p) vf_st(p, v)
 *   vf_add vf_sub vf_mul vf_div vf_sqrt   (IEEE, no FMA contraction)
 *   vm vf_gt(a, b) vm vf_ge(a, b)         a > b, a >= b (false for NaN)
 *   vf vf_sel(m, a, b)                    m ? a : b
 *   vf vf_floor(v)                        exact floor for |v| < 2^31
 *   vf vf_exp_of(v)                       unbiased exponent of v (as float)
 *   vf vf_mant_of(v)                      mantissa of v in [1, 2)
 *   vf vf_pow2i(v)                        2^v for integral v in [-126, 127]
 */

#if FM_FEATURE_TNL

FM_INLINE vf fmf_max0(vf x) { return vf_sel(vf_gt(x, vf_set(0.0f)), x, vf_set(0.0f)); }
FM_INLINE vf fmf_clamp01(vf x) { return vf_sel(vf_gt(x, vf_set(0.0f)), vf_sel(vf_gt(vf_set(1.0f), x), x, vf_set(1.0f)), vf_set(0.0f)); }
FM_INLINE vf fmf_dot3(vf ax, vf ay, vf az, vf bx, vf by, vf bz)
{
    return vf_add(vf_add(vf_mul(ax, bx), vf_mul(ay, by)), vf_mul(az, bz));
}

/* log2 for x > 0 (normal floats): exponent + atanh series of (m-1)/(m+1),
 * |error| < 5e-6 */
FM_INLINE vf fmf_log2(vf x)
{
    vf m  = vf_mant_of(x);
    vf s  = vf_div(vf_sub(m, vf_set(1.0f)), vf_add(m, vf_set(1.0f)));
    vf s2 = vf_mul(s, s);
    vf p  = vf_add(vf_set(0.9617966939f), vf_mul(s2, vf_add(vf_set(0.5770780164f), vf_mul(s2, vf_set(0.4121985832f)))));
    p     = vf_mul(s, vf_add(vf_set(2.8853900818f), vf_mul(s2, p)));
    return vf_add(vf_exp_of(x), p);
}

/* 2^y, y clamped to [-126, 126]; 7 term series for the fraction */
FM_INLINE vf fmf_exp2(vf y)
{
    y    = vf_sel(vf_gt(y, vf_set(-126.0f)), y, vf_set(-126.0f));
    y    = vf_sel(vf_gt(vf_set(126.0f), y), y, vf_set(126.0f));
    vf i = vf_floor(y);
    vf f = vf_sub(y, i);
    vf p = vf_add(vf_set(0.0096181291f), vf_mul(f, vf_add(vf_set(0.0013333558f), vf_mul(f, vf_set(0.0001540353f)))));
    p    = vf_add(vf_set(0.2402265070f), vf_mul(f, vf_add(vf_set(0.0555041087f), vf_mul(f, p))));
    p    = vf_add(vf_set(1.0f), vf_mul(f, vf_add(vf_set(0.6931471806f), vf_mul(f, p))));
    return vf_mul(p, vf_pow2i(i));
}

/* x^e for x in [0, 1] (0 for x <= tiny) */
FM_INLINE vf fmf_pow01(vf x, float e)
{
    vf pos = vf_sel(vf_gt(x, vf_set(1e-30f)), x, vf_set(1.0f));
    vf r   = fmf_exp2(vf_mul(vf_set(e), fmf_log2(pos)));
    return vf_sel(vf_gt(x, vf_set(1e-30f)), r, vf_set(0.0f));
}

FM_INLINE vf fmf_rlen(vf x, vf y, vf z) /* 1 / |v| (0 vectors: huge, harmless) */
{
    vf d = fmf_dot3(x, y, z, x, y, z);
    return vf_div(vf_set(1.0f), vf_sqrt(vf_sel(vf_gt(d, vf_set(1e-30f)), d, vf_set(1e-30f))));
}

/* one vector of vertices, see fm_light_params (fm_internal.h) */
FM_INLINE void fmf_light_vec(const fm_light_params* p, const float* in[10], float* out[4], int i)
{
    vf px = vf_ld(in[0] + i), py = vf_ld(in[1] + i), pz = vf_ld(in[2] + i);
    vf nx = vf_ld(in[3] + i), ny = vf_ld(in[4] + i), nz = vf_ld(in[5] + i);
    vf r  = fmf_rlen(nx, ny, nz);
    nx = vf_mul(nx, r), ny = vf_mul(ny, r), nz = vf_mul(nz, r);
    /* local viewer: V = normalize(-P) */
    vf vx = vf_sub(vf_set(0.0f), px), vy = vf_sub(vf_set(0.0f), py), vz = vf_sub(vf_set(0.0f), pz);
    r     = fmf_rlen(vx, vy, vz);
    vx = vf_mul(vx, r), vy = vf_mul(vy, r), vz = vf_mul(vz, r);
    vf ac[3], dc[3], sc[3];
    for (int k = 0; k < 3; k++) ac[k] = dc[k] = sc[k] = vf_set(0.0f);
    for (int l = 0; l < p->nlights; l++) {
        const struct fm_light_k* L = &p->l[l];
        vf lx, ly, lz, att;
        if (L->type == 0) { /* directional: dir points towards the light */
            lx = vf_set(L->dir[0]), ly = vf_set(L->dir[1]), lz = vf_set(L->dir[2]);
            att = vf_set(1.0f);
        } else {
            vf dx = vf_sub(vf_set(L->pos[0]), px), dy = vf_sub(vf_set(L->pos[1]), py), dz = vf_sub(vf_set(L->pos[2]), pz);
            vf d2 = fmf_dot3(dx, dy, dz, dx, dy, dz);
            vf d  = vf_sqrt(vf_sel(vf_gt(d2, vf_set(1e-30f)), d2, vf_set(1e-30f)));
            vf id = vf_div(vf_set(1.0f), d);
            lx = vf_mul(dx, id), ly = vf_mul(dy, id), lz = vf_mul(dz, id);
            att = vf_div(vf_set(1.0f), vf_add(vf_add(vf_set(L->katt[0]), vf_mul(vf_set(L->katt[1]), d)),
                                              vf_mul(vf_set(L->katt[2]), d2)));
            if (L->type == 2) { /* spot: dir is the cone axis (light -> scene) */
                vf sd = vf_sub(vf_set(0.0f), fmf_dot3(lx, ly, lz, vf_set(L->dir[0]), vf_set(L->dir[1]), vf_set(L->dir[2])));
                vf sf = L->spot_exp > 0.0f ? fmf_pow01(fmf_max0(sd), L->spot_exp) : vf_set(1.0f);
                att   = vf_mul(att, vf_sel(vf_ge(sd, vf_set(L->spot_cos)), sf, vf_set(0.0f)));
            }
        }
        vf ndl = fmf_max0(fmf_dot3(nx, ny, nz, lx, ly, lz));
        vf hx = vf_add(lx, vx), hy = vf_add(ly, vy), hz = vf_add(lz, vz);
        r      = fmf_rlen(hx, hy, hz);
        vf ndh = fmf_max0(fmf_dot3(nx, ny, nz, vf_mul(hx, r), vf_mul(hy, r), vf_mul(hz, r)));
        vf sp  = vf_sel(vf_gt(ndl, vf_set(0.0f)), fmf_pow01(ndh, p->shininess), vf_set(0.0f));
        vf ad  = vf_mul(att, ndl), as = vf_mul(att, sp);
        for (int k = 0; k < 3; k++) {
            ac[k] = vf_add(ac[k], vf_mul(att, vf_set(L->amb[k])));
            dc[k] = vf_add(dc[k], vf_mul(ad, vf_set(L->dif[k])));
            sc[k] = vf_add(sc[k], vf_mul(as, vf_set(L->spe[k])));
        }
    }
    for (int k = 0; k < 3; k++) {
        vf md = p->color_material ? vf_ld(in[6 + k] + i) : vf_set(p->mat_dif[k]);
        vf ma = p->color_material ? md : vf_set(p->mat_amb[k]);
        vf c  = vf_add(vf_set(p->mat_emi[k]), vf_mul(vf_set(p->gamb[k]), ma));
        c     = vf_add(c, vf_mul(ac[k], ma));
        c     = vf_add(c, vf_mul(dc[k], md));
        c     = vf_add(c, vf_mul(sc[k], vf_set(p->mat_spe[k])));
        vf_st(out[k] + i, fmf_clamp01(c));
    }
    vf_st(out[3] + i, fmf_clamp01(p->color_material ? vf_ld(in[9] + i) : vf_set(p->mat_dif[3])));
}

static void FMK(light)(const fm_light_params* p, const float* const* in, int n, float* const* out)
{
    const float* ip[10];
    float*       op[4];
    for (int k = 0; k < 10; k++) ip[k] = in[k];
    for (int k = 0; k < 4; k++) op[k] = out[k];
    int i = 0;
    for (; i + FMF_W <= n; i += FMF_W) fmf_light_vec(p, ip, op, i);
    if (i < n) { /* tail: padded copies, the lanes are independent */
        float tin[10][FMF_W], tout[4][FMF_W];
        memset(tin, 0, sizeof(tin));
        for (int k = 0; k < 10; k++) {
            for (int j = 0; j < n - i; j++) tin[k][j] = in[k][i + j];
            ip[k] = tin[k];
        }
        for (int k = 0; k < 4; k++) op[k] = tout[k];
        fmf_light_vec(p, ip, op, 0);
        for (int k = 0; k < 4; k++)
            for (int j = 0; j < n - i; j++) out[k][i + j] = tout[k][j];
    }
}

#endif

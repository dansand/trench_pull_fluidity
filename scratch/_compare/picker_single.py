def pick_trench_3step(
    x, z, p, vx,
    p_depth_m=0.0,            # depth for pressure row (0 -> z[0])
    vx_depth_m=5_000.0,       # depth for velocity diagnostics
    Wp1_km=800.0,             # coarse pressure window
    Wv_km=800.0,              # velocity window
    Wp3_km=300.0,             # FINAL pressure window (directional)
    expected_offset_km=50.0,  # expected trench offset from velocity transition
    smooth_km=20.0,           # smoothing ONLY for velocity
    subducting_side="left"    # "left" or "right" of velocity transition
):
    """
    Three-step non-stateful trench picker:
      1) coarse surface pressure minimum
      2) velocity sign-change (plate boundary)
      3) directional, windowed TRUE pressure minimum
    """

    # ---------------------------
    # Select depth rows
    # ---------------------------
    iz_p  = 0 if p_depth_m == 0.0 else int(np.argmin(np.abs(z - p_depth_m)))
    iz_vx = int(np.argmin(np.abs(z - vx_depth_m)))

    p_row  = p[iz_p, :].copy()
    vx_row = vx[iz_vx, :].copy()

    # ---------------------------
    # STEP 1: coarse pressure minimum
    # ---------------------------
    j0 = int(np.nanargmin(p_row))
    x_p0 = float(x[j0])

    # ---------------------------
    # STEP 2: velocity transition
    # ---------------------------
    # ---------------------------
    # STEP 2: velocity transition
    # ---------------------------
    Wv = Wv_km * 1000.0
    in_v = (x >= x_p0 - Wv) & (x <= x_p0 + Wv)
    
    x_vw  = x[in_v]
    vx_vw = vx_row[in_v]
    
    if x_vw.size < 5:
        raise RuntimeError("Velocity window too small")
    
    # smooth velocity ONLY
    dx = np.median(np.diff(x))
    k = max(1, int((smooth_km * 1000.0) / dx))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    vx_s = np.convolve(vx_vw, kernel, mode="same")
    
    s = np.sign(vx_s)
    s[s == 0] = np.nan
    cross = np.where(
        np.isfinite(s[:-1]) &
        np.isfinite(s[1:]) &
        (s[:-1] * s[1:] < 0)
    )[0]
    
    if cross.size == 0:
        jv = int(np.nanargmin(np.abs(vx_s)))
        x_v = float(x_vw[jv])
        vx_method = "min|vx|"
    else:
        xc = []
        for i in cross:
            x1, x2 = x_vw[i], x_vw[i+1]
            v1, v2 = vx_s[i], vx_s[i+1]
            xc.append(x1 - v1 * (x2 - x1) / (v2 - v1))
        xc = np.array(xc)
    
        # 🔑 NEW RULE: pick leftmost velocity transition
        x_v = float(np.min(xc))
        vx_method = "zero_cross_leftmost"

    # ---------------------------
    # STEP 3: directional, windowed TRUE pressure minimum
    # ---------------------------
    Wp3 = Wp3_km * 1000.0
    expected_offset = expected_offset_km * 1000.0

    if subducting_side == "left":
        mask = (x < x_v) & (x >= x_v - Wp3)
        x_expected = x_v - expected_offset
    else:
        mask = (x > x_v) & (x <= x_v + Wp3)
        x_expected = x_v + expected_offset

    xw = x[mask]
    pw = p_row[mask]

    if xw.size < 5:
        raise RuntimeError("Pressure window too small")

    # remove NaNs
    good = np.isfinite(pw)
    xw, pw = xw[good], pw[good]

    if xw.size < 5:
        raise RuntimeError("Too many NaNs in pressure window")

    # TRUE local minima (no smoothing)
    is_min = (pw[1:-1] < pw[:-2]) & (pw[1:-1] < pw[2:])
    idx = np.where(is_min)[0] + 1

    if idx.size == 0:
        # explicit fallback
        j = int(np.argmin(pw))
        trench_x = float(xw[j])
        p_method = "fallback_global_min"
    else:
        # choose minimum closest to expected offset
        j = idx[np.argmin(np.abs(xw[idx] - x_expected))]
        trench_x = float(xw[j])
        p_method = "local_min_windowed"

    # ---------------------------
    # Diagnostics
    # ---------------------------
    info = {
        "x_p0": x_p0,
        "x_v": x_v,
        "trench_x": trench_x,
        "vx_method": vx_method,
        "p_method": p_method,
        "subducting_side": subducting_side,
        "depths_m": {
            "pressure": z[iz_p],
            "velocity": z[iz_vx]
        },
        "windows_km": {
            "Wp1": Wp1_km,
            "Wv": Wv_km,
            "Wp3": Wp3_km
        },
        "debug": {
            "x_vw": x_vw,
            "vx_s": vx_s,
            "xw": xw,
            "pw": pw,
            "x_candidates": xw[idx] if idx.size else None,
            "p_candidates": pw[idx] if idx.size else None
        }
    }

    return trench_x, info


# ## 4. Build grid and extract fields (single timestep)

def pick_trench_3step(x, z, p, vx,
        p_depth_m=0.0, vx_depth_m=5_000.0,
        Wp1_km=800.0, Wv_km=800.0, Wp3_km=300.0,
        expected_offset_km=50.0, smooth_km=20.0,
        subducting_side='left'):
    iz_p  = 0 if p_depth_m == 0.0 else int(np.argmin(np.abs(z - p_depth_m)))
    iz_vx = int(np.argmin(np.abs(z - vx_depth_m)))
    p_row, vx_row = p[iz_p, :].copy(), vx[iz_vx, :].copy()

    j0 = int(np.nanargmin(p_row));  x_p0 = float(x[j0])

    Wv = Wv_km * 1000.0
    in_v = (x >= x_p0 - Wv) & (x <= x_p0 + Wv)
    x_vw, vx_vw = x[in_v], vx_row[in_v]
    dx = np.median(np.diff(x))
    k = max(1, int((smooth_km*1000.0)/dx)); k += (k%2 == 0)
    vx_s = np.convolve(vx_vw, np.ones(k)/k, mode='same')
    s = np.sign(vx_s); s[s == 0] = np.nan
    cross = np.where(np.isfinite(s[:-1]) & np.isfinite(s[1:]) & (s[:-1]*s[1:] < 0))[0]
    if cross.size == 0:
        x_v = float(x_vw[int(np.nanargmin(np.abs(vx_s)))])
    else:
        xc = []
        for ii in cross:
            x1, x2 = x_vw[ii], x_vw[ii+1]
            v1, v2 = vx_s[ii], vx_s[ii+1]
            xc.append(x1 - v1 * (x2 - x1) / (v2 - v1))
        x_v = float(np.min(np.array(xc)))

    Wp3 = Wp3_km * 1000.0
    expected_offset = expected_offset_km * 1000.0
    if subducting_side == 'left':
        mask = (x < x_v) & (x >= x_v - Wp3)
        x_expected = x_v - expected_offset
    else:
        mask = (x > x_v) & (x <= x_v + Wp3)
        x_expected = x_v + expected_offset
    xw, pw = x[mask], p_row[mask]
    good = np.isfinite(pw); xw, pw = xw[good], pw[good]
    is_min = (pw[1:-1] < pw[:-2]) & (pw[1:-1] < pw[2:])
    idx = np.where(is_min)[0] + 1
    if idx.size == 0:
        trench_x = float(xw[int(np.argmin(pw))])
    else:
        j = idx[np.argmin(np.abs(xw[idx] - x_expected))]
        trench_x = float(xw[j])

    return trench_x, dict(x_p0=x_p0, x_v=x_v)


# In[ ]:

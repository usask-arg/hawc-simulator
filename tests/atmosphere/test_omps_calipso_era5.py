from __future__ import annotations

import numpy as np

from hawcsimulator.atmosphere import ompscalera


def test_load_data():
    ds = ompscalera.load_data(30.0)

    np.testing.assert_allclose(ds["altitude"], np.arange(0, 40.0, 0.5))
    assert float(ds["latitude"]) == 30.0

    # The OMPS aerosol is only defined in the stratosphere
    stratosphere = (ds["altitude"] >= 15) & (ds["altitude"] <= 30)
    extinction = ds["extinction"].where(stratosphere, drop=True)
    assert np.isfinite(extinction).all()
    assert (extinction > 0).all()

    h2o_vmr = ds["h2o_vmr"]
    assert np.isfinite(h2o_vmr).all()
    assert (h2o_vmr > 0).all()
    # Wet near the surface, a few ppm in the stratosphere
    assert h2o_vmr.sel(altitude=0.0) > 1e-3
    assert 1e-6 < h2o_vmr.sel(altitude=25.0) < 2e-5

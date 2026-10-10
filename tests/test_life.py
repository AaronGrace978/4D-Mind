import numpy as np

from tesseract.life import (
    OCC,
    SCAR,
    TISSUE,
    alive_frames,
    load_iris,
    occupancy_com,
    rasterize,
    truncate_invariants,
)


def test_iris_ends_in_the_same_room():
    volume, _inv, meta = rasterize(load_iris(), (16, 32, 32, 32))
    alive = alive_frames(volume)
    start = occupancy_com(volume, int(alive[0]))
    end = occupancy_com(volume, int(alive[-1]))
    assert meta["same_room"] is True
    assert np.linalg.norm(start - end) < 0.08
    assert abs(start[0] - meta["room"][0]) < 0.08
    assert abs(end[0] - meta["room"][0]) < 0.08


def test_scar_opens_and_stays():
    spec = load_iris()
    volume, inv, _meta = rasterize(spec, (16, 32, 32, 32))
    times = (np.arange(volume.shape[0]) + 0.5) / volume.shape[0]
    before = volume[times < spec["scar_t"] - 0.03][..., SCAR].sum()
    after = volume[(times > spec["scar_t"] + 0.04) & (times < spec["lifespan"])][..., SCAR].sum()
    assert before == 0
    assert after > 0
    assert inv[1] > 0.3


def test_tissue_is_inside_the_body_and_worldlines_meet():
    volume, inv, _meta = rasterize(load_iris(), (16, 32, 32, 32))
    tissue = volume[..., TISSUE] > 0.5
    occupied = volume[..., OCC] > 0.3
    assert tissue.sum() > 0
    assert (tissue & occupied).sum() / tissue.sum() > 0.8
    assert ((volume[..., 4] > 0.3) & occupied).sum() > 0
    assert inv[2] > 0


def test_truncate_hides_a_future_scar():
    inv = np.array([0.9, 0.7, 0.4, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.2], dtype=np.float32)
    hidden = truncate_invariants(inv, visible_end=0.5)
    assert hidden[1] == -1
    assert hidden[2] > 0
    assert hidden[0] == np.float32(0.5)

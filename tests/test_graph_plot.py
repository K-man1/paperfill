"""
Tests for plotting a model's graph answer onto a detected coordinate grid.

The mapping is the part with real consequences: an off-by-one in the origin or
a flipped axis puts a whole parabola in the wrong quadrant, and nothing
downstream would notice. Grid numbers here are the ones detected off the
Precalculus packet's graphs (a -10..10 grid, 217pt wide, origin at its centre).
"""

import pytest

from paperfill.ai.render import curve_through, parse_points, plot_point

GRID = {
    "left": 337.0, "right": 554.0, "top": 127.0, "bottom": 343.0,
    "origin_x": 445.0, "origin_y": 235.0,
    "x_min": -10.0, "x_max": 10.0, "y_min": -10.0, "y_max": 10.0,
}


@pytest.mark.parametrize("answer, expected", [
    ("(-1, 7), (0, 1), (1, -1)", [(-1, 7), (0, 1), (1, -1)]),
    ("The points are (0,1) and (2,1).", [(0, 1), (2, 1)]),
    ("(-0.5, 3.5)", [(-0.5, 3.5)]),
    ("[(-1, 7), (0, 1)]", [(-1, 7), (0, 1)]),
    # Real minus sign and en dash, which models emit when writing "maths".
    ("(−1, 7), (0, −1)", [(-1, 7), (0, -1)]),
    ("no points here", []),
    ("", []),
])
def test_parse_points(answer, expected):
    assert parse_points(answer) == expected


def test_json_list_answer_survives_being_flattened():
    """A model answering with a nested list gets flattened to a bare run of
    numbers with every bracket stripped before the renderer sees it. That used
    to plot nothing at all."""
    assert parse_points("-1, 7, 0, 1, 1, -1") == [(-1, 7), (0, 1), (1, -1)]


def test_an_odd_run_of_loose_numbers_is_not_guessed_at():
    assert parse_points("1, 2, 3") == []


def test_origin_maps_to_the_detected_axis_crossing():
    assert plot_point(GRID, 0, 0) == (GRID["origin_x"], GRID["origin_y"])


def test_y_grows_upward_on_the_page():
    """PDF y increases downward, so a positive graph y must come out ABOVE the
    origin. Getting this backwards mirrors every plot through the x-axis."""
    _, above = plot_point(GRID, 0, 5)
    _, below = plot_point(GRID, 0, -5)
    assert above < GRID["origin_y"] < below


def test_unit_scale_matches_the_detected_grid():
    x_at_1, _ = plot_point(GRID, 1, 0)
    x_at_2, _ = plot_point(GRID, 2, 0)
    per_unit = (GRID["right"] - GRID["left"]) / 20
    assert x_at_1 - GRID["origin_x"] == pytest.approx(per_unit)
    assert x_at_2 - x_at_1 == pytest.approx(per_unit)


@pytest.mark.parametrize("x, y", [(0, 40), (0, -40), (40, 0), (-40, 0)])
def test_points_off_the_grid_are_dropped(x, y):
    """A parabola sampled across the full x-range leaves the visible y-range
    almost immediately; those points must not be clamped to the edge, which
    would draw a false flat line along the top of the plot."""
    assert plot_point(GRID, x, y) is None


def test_degenerate_axis_range_is_refused():
    assert plot_point({**GRID, "x_min": 0.0, "x_max": 0.0}, 1, 1) is None


PARABOLA = [(x / 2, (x / 2) ** 2) for x in range(-6, 7)]


def test_curve_passes_through_every_control_point():
    """Catmull-Rom was chosen over a smoothing spline precisely because the
    points ARE the answer: a curve that only approximates them plots something
    the model never said."""
    curve = curve_through(PARABOLA)
    for point in PARABOLA:
        assert any(abs(cx - point[0]) < 1e-6 and abs(cy - point[1]) < 1e-6
                   for cx, cy in curve), f"{point} missing from the curve"


def test_curve_is_densified_and_ordered():
    curve = curve_through(PARABOLA)
    assert len(curve) > len(PARABOLA) * 8
    assert curve == sorted(curve), "curve must run left to right"


def test_curve_stays_within_the_data_range():
    """The padded end segments exist so the curve stops at the outermost point
    instead of overshooting off the edge of the grid."""
    curve = curve_through(PARABOLA)
    xs = [x for x, _ in curve]
    assert min(xs) == pytest.approx(-3.0)
    assert max(xs) == pytest.approx(3.0)


@pytest.mark.parametrize("points", [[], [(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)]])
def test_too_few_points_to_curve_are_left_alone(points):
    assert curve_through(points) == points

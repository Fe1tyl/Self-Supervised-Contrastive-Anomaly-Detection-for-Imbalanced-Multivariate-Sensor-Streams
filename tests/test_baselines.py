import numpy as np

from hai_repro.baselines import _window_summary_features


def test_window_summary_features_are_aligned_and_ordered() -> None:
    values = np.asarray(
        [
            [0.0, 10.0],
            [1.0, 8.0],
            [2.0, 6.0],
            [3.0, 4.0],
            [4.0, 2.0],
        ],
        dtype=np.float32,
    )
    endpoints, features = _window_summary_features(
        values, length=3, stride=2, batch_size=1
    )

    np.testing.assert_array_equal(endpoints, np.asarray([2, 4]))
    assert features.shape == (2, 12)
    first_window = values[:3]
    expected = np.concatenate(
        (
            first_window.mean(axis=0),
            first_window.std(axis=0),
            first_window.min(axis=0),
            first_window.max(axis=0),
            first_window[-1],
            first_window[-1] - first_window[0],
        )
    )
    np.testing.assert_allclose(features[0], expected, atol=1e-6)


def test_window_summary_features_reject_nonfinite_output() -> None:
    values = np.asarray([[0.0], [np.nan], [2.0]], dtype=np.float32)
    try:
        _window_summary_features(values, length=2, stride=1, batch_size=2)
    except FloatingPointError:
        return
    raise AssertionError("Expected non-finite window features to be rejected")

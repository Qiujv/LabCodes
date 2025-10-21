import numpy as np
import pytest

from labcodes import state_disc


def test_probs_from_flags_multiqubit_with_labels():
    flags = [[1, 1, 0, 0], [1, 0, 1, 0]]
    probs, labels = state_disc.probs_from_flags(
        flags, nlevels=2, n_qbs=2, return_labels=True
    )

    expected_prob = np.full(4, 0.25)
    expected_labels = np.array(["00", "01", "10", "11"])

    assert np.allclose(probs, expected_prob)
    assert np.array_equal(labels, expected_labels)

    assert np.array_equal(
        ["000", "001", "010", "011", "100", "101", "110", "111"],
        state_disc.prob_labels(n_qbs=3, nlevels=2),
    )


@pytest.mark.parametrize(
    "flags, n_qbs, nlevels, expected",
    [
        (np.arange(4), 2, 2, np.array(["00", "01", "10", "11"])),
        ([5, 6, 7], 3, 2, np.array(["101", "110", "111"])),
    ],
)
def test_str_from_flags_and_back(flags, n_qbs, nlevels, expected):
    strings = state_disc.str_from_flags(flags, n_qbs=n_qbs, nlevels=nlevels)
    round_trip = state_disc.flags_from_str(strings, nlevels=nlevels)

    assert np.array_equal(strings, expected)
    assert np.array_equal(round_trip, np.asarray(flags))


def test_flags_mq_from_1q_matches_example():
    list_flags = [
        [0, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 1],
    ]
    combined = state_disc.flags_mq_from_1q(list_flags, nlevels=2)

    expected = np.array([0, 1, 2, 4, 6, 5, 3, 7])

    assert np.array_equal(combined, expected)


def test_ncenter_fit_recovers_high_classification_accuracy():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]])
    list_pts = [rng.normal(loc=center, scale=0.1, size=(200, 2)) for center in centers]

    stater = state_disc.NCenter.fit(list_pts)

    assert stater.centers.shape == centers.shape

    for idx, pts in enumerate(list_pts):
        flags = stater.flags(pts)
        probs = stater.probs(pts)

        accuracy = np.mean(flags == idx)

        assert accuracy > 0.9
        assert probs.shape == (len(centers),)
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[idx] > 0.9
        assert np.isclose(probs[idx], accuracy, atol=1e-6)

"""Automated tests for the customer segmentation Flask app."""

import math

import pytest

import app as segmentation_app


@pytest.fixture()
def client():
    segmentation_app.app.config["TESTING"] = True
    return segmentation_app.app.test_client()


def predict(client, income, score):
    return client.post(
        "/predict",
        data={"income": income, "score": score},
        follow_redirects=True,
    )


def segment_names():
    return [d["name"] for d in segmentation_app.CLUSTER_DETAILS.values()]


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.parametrize(
    "income, score, expected_segment",
    [
        (90, 90, "High-Income, High-Spending Customers"),
        (90, 10, "High-Income, Low-Spending Customers"),
        (20, 90, "Lower-Income, High-Spending Customers"),
        (20, 10, "Lower-Income, Low-Spending Customers"),
        (55, 50, "Moderate-Income, Moderate-Spending Customers"),
    ],
)
def test_prediction_returns_expected_segment(client, income, score, expected_segment):
    response = predict(client, income, score)
    assert response.status_code == 200
    assert expected_segment in response.get_data(as_text=True)


@pytest.mark.parametrize(
    "income, score",
    [
        ("", ""),
        ("", 50),
        (50, ""),
        ("abc", 50),
        (50, "abc"),
        (-5, 50),
        (50, 0),
        (50, 101),
        ("nan", 50),
        (50, "inf"),
    ],
)
def test_invalid_input_is_rejected(client, income, score):
    response = predict(client, income, score)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "flash-message" in html
    assert not any(name in html for name in segment_names())


def test_every_cluster_has_display_details():
    model = segmentation_app.model
    assert model is not None
    for cluster_id in range(model.n_clusters):
        details = segmentation_app.CLUSTER_DETAILS[cluster_id]
        assert details["name"] and details["advice"]


def test_segment_names_match_cluster_centers():
    """Guards against mislabeled clusters (K-Means IDs are arbitrary)."""
    model = segmentation_app.model
    scaler = segmentation_app.scaler
    centers = scaler.inverse_transform(model.cluster_centers_)

    for cluster_id, (income, score) in enumerate(centers):
        name = segmentation_app.CLUSTER_DETAILS[cluster_id]["name"]

        if name.startswith("High-Income"):
            assert income > 70, f"{name}: income center {income:.1f}"
        elif name.startswith("Moderate-Income"):
            assert 40 <= income <= 70, f"{name}: income center {income:.1f}"
        elif name.startswith("Lower-Income"):
            assert income < 40, f"{name}: income center {income:.1f}"
        else:
            pytest.fail(f"Unexpected income label in {name}")

        if "High-Spending" in name:
            assert score > 60, f"{name}: spending center {score:.1f}"
        elif "Moderate-Spending" in name:
            assert 40 <= score <= 60, f"{name}: spending center {score:.1f}"
        elif "Low-Spending" in name:
            assert score < 40, f"{name}: spending center {score:.1f}"
        else:
            pytest.fail(f"Unexpected spending label in {name}")

        assert math.isfinite(income) and math.isfinite(score)

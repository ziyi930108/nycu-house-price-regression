"""Target-free administrative-address correction for round-three experiments.

The frozen house_features module and its 87-column schema stay unchanged.
Only a supplied district that exactly matches the address prefix is trusted.
Without that evidence the existing parsing is retained; no external place
dictionary, price information, or data-fitted vocabulary is used.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from house_features import build_features, _text, _cat, _source, _CN, MISSING

ADDRESS_COLUMNS = ("address_district", "address_road", "address_section",
                   "address_locality", "location_road")


def correct_address_features(raw: pd.DataFrame, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return an independent feature frame; do not alter raw or X in place.

    X may be the old cached features for these exact raw rows. Numeric features,
    all non-address categories, column order and column dtypes are preserved.
    Old base models should continue to receive X, while a separately validated
    reference candidate may receive this corrected copy.
    """
    base = build_features(raw) if X is None else X
    if len(base) != len(raw) or not base.index.equals(raw.index):
        raise ValueError("raw and X must have identical row count and index.")
    if "編號" in base or "總價元" in base:
        raise ValueError("X must contain model features only, without ID/target.")
    if not set(ADDRESS_COLUMNS).issubset(base.columns):
        raise ValueError("X does not contain the expected address feature columns.")
    result = base.copy(deep=True)
    address = _text(_source(raw, "土地位置建物門牌")).str.replace("台", "臺", regex=False)
    city = address.str.extract(r"^(.{2,3}?[縣市])", expand=False)
    after_city = address.str.replace(r"^.{2,3}?[縣市]", "", regex=True)
    original_parsed_district = after_city.str.extract(r"^(.{1,4}?[區鄉鎮市])", expand=False)
    supplied = _text(_source(raw, "鄉鎮市區")).str.replace("台", "臺", regex=False)
    supplied_valid = supplied.str.fullmatch(r".{1,5}[區鄉鎮市]", na=False)
    prefix_matches = pd.Series(
        [bool(ok and pd.notna(a) and pd.notna(d) and a.startswith(d))
         for a, d, ok in zip(after_city, supplied, supplied_valid)],
        index=raw.index, dtype=bool,
    )
    chosen = original_parsed_district.where(~prefix_matches, supplied)
    # Retain the old suffix removal exactly when the supplied district gives no evidence.
    rest = after_city.str.replace(r"^.{1,4}?[區鄉鎮市]", "", regex=True)
    if prefix_matches.any():
        rest.loc[prefix_matches] = [
            a[len(d):] for a, d in zip(after_city.loc[prefix_matches], supplied.loc[prefix_matches])
        ]
    road = rest.str.extract(r"^(.{1,18}?(?:大道|路|街))", expand=False)
    values = {
        "address_district": _cat(chosen.fillna(_text(_source(raw, "鄉鎮市區")))),
        "address_road": _cat(road),
        "address_section": _cat(rest.str.extract(r"([0-9" + _CN + r"]+段)", expand=False)),
        "address_locality": _cat(rest.str.extract(r"^(.{1,12}?(?:段|里|村))", expand=False)),
        "location_road": _cat(city.fillna("") + "|" +
                              chosen.fillna(base["district"]) + "|" + road.fillna(MISSING)),
    }
    for name, value in values.items():
        result[name] = value.astype(base[name].dtype)
    if list(result.columns) != list(base.columns) or not result.dtypes.equals(base.dtypes):
        raise AssertionError("Feature schema changed.")
    return result


def corrected_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed legacy schema and apply the address-only correction."""
    return correct_address_features(raw)


def self_check():
    raw = pd.DataFrame({
        "編號": np.arange(1, 9),
        "總價元": np.arange(1, 9) * 1_000_000,
        "鄉鎮市區": ["新市區", "前鎮區", "平鎮區", "大安區", "竹北市", "二林鎮", "中西區", None],
        "土地位置建物門牌": [
            "臺南市新市區中正路一段10號", "高雄市前鎮區一心一路10號",
            "桃園市平鎮區中豐路二段10號", "臺北市大安區市民大道四段10號",
            "新竹縣竹北市文興路一段10號", "彰化縣二林鎮斗苑路五段10號",
            "臺南市中西區民生路二段10號", "臺南市新市區中正路10號",
        ],
        "交易年月日": ["1130811"] * 8,
    })
    original_raw = raw.copy(deep=True)
    old = build_features(raw)
    original_old = old.copy(deep=True)
    corrected = correct_address_features(raw, old)
    assert corrected.address_district.iloc[:7].tolist() == raw["鄉鎮市區"].iloc[:7].tolist()
    assert corrected.address_road.iloc[:7].tolist() == [
        "中正路", "一心一路", "中豐路", "市民大道", "文興路", "斗苑路", "民生路"]
    assert corrected.address_section.iloc[:7].tolist() == ["一段", MISSING, "二段", "四段", "一段", "五段", "二段"]
    assert corrected.loc[7, "address_district"] == old.loc[7, "address_district"]
    untouched = [name for name in old if name not in ADDRESS_COLUMNS]
    pd.testing.assert_frame_equal(corrected[untouched], old[untouched])
    pd.testing.assert_frame_equal(raw, original_raw)
    pd.testing.assert_frame_equal(old, original_old)
    alternate = raw.copy(deep=True)
    alternate["編號"] += 100
    alternate["總價元"] *= 17
    pd.testing.assert_frame_equal(corrected, corrected_features(alternate))
    assert len(corrected.columns) == 87 and corrected.dtypes.equals(old.dtypes)
    assert "編號" not in corrected and "總價元" not in corrected
    return {"status": "passed", "rows": len(raw), "columns": len(corrected.columns),
            "numeric_features_unchanged": True, "input_unchanged": True, "target_and_id_independent": True}


if __name__ == "__main__":
    print(json.dumps(self_check(), ensure_ascii=False))

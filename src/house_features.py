"""Deterministic, target-free features for Taiwan real-estate transactions.

Apply the same function to train, validation and test. No vocabulary or target
statistic is learned. Numeric outputs are float32 (NaN allowed); categories are
strings. Prices, row identifiers, raw remarks and cancellation status are unused.
"""
from __future__ import annotations

from functools import lru_cache
import re
import numpy as np
import pandas as pd

MISSING = "__MISSING__"
_DIGITS = {c: i for i, c in enumerate("零一二三四五六七八九")}
_DIGITS.update({"〇": 0, "○": 0, "兩": 2, "壹": 1, "貳": 2, "參": 3,
                "肆": 4, "伍": 5, "陸": 6, "柒": 7, "捌": 8, "玖": 9})
_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000}
_CN = "零〇○一二三四五六七八九十百千兩壹貳參肆伍陸柒捌玖拾佰"
_FLOOR_RE = re.compile(r"(?P<prefix>地下|地上|負|B|-)?\s*(?P<number>[0-9" + _CN + r"]+)", re.I)

def _source(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df:
            return df[name]
    return pd.Series(np.nan, index=df.index)

def _text(s: pd.Series) -> pd.Series:
    result = s.astype("string").str.strip().str.replace("\u3000", "", regex=False)
    return result.mask(result.isin(["", "nan", "NaN", "None", "null", "<NA>"]))

def _cat(s: pd.Series) -> pd.Series:
    return _text(s).fillna(MISSING).astype(object)

def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_text(s).str.replace(",", "", regex=False), errors="coerce").astype("float32")

@lru_cache(maxsize=2048)
def _chinese_number(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    if not value or any(c not in _DIGITS and c not in _UNITS for c in value):
        return np.nan
    if not any(c in _UNITS for c in value):
        return float("".join(str(_DIGITS[c]) for c in value))
    total, current = 0, 0
    for c in value:
        if c in _DIGITS:
            current = _DIGITS[c]
        else:
            total += (current or 1) * _UNITS[c]
            current = 0
    return float(total + current)

@lru_cache(maxsize=16384)
def _floor_tuple(value: str) -> tuple[float, ...]:
    if value == MISSING:
        return (np.nan,) * 5 + (0.0, 0.0, 0.0)
    value = value.upper().replace("－", "-").replace("、", ",")
    value = re.sub(r"(?<=[0-9" + _CN + r"])\s*-\s*(?=[0-9" + _CN + r"])", "至", value)
    floors = []
    for match in _FLOOR_RE.finditer(value):
        number = _chinese_number(match.group("number"))
        if np.isfinite(number):
            prefix = match.group("prefix") or ""
            floors.append(-number if prefix in {"地下", "負", "B", "-"} else number)
    floors = sorted(set(floors))
    all_building = float(bool(re.search(r"全[層部棟]?|整棟", value)))
    rooftop = float(bool(re.search(r"屋頂|突出|頂樓", value)))
    basement = float("地下" in value or bool(re.search(r"B\d", value)) or any(f < 0 for f in floors))
    if not floors:
        return (np.nan,) * 5 + (basement, all_building, rooftop)
    count = float(len(floors))
    if len(floors) == 2 and re.search(r"至|到|~|～", value):
        count = abs(floors[1] - floors[0]) + 1
        if floors[0] < 0 < floors[1]:
            count -= 1  # Floor zero does not exist between basement and ground floors.
    return (float(floors[0]), float(floors[-1]), float(np.mean(floors)), count,
            float(floors[-1] - floors[0]), basement, all_building, rooftop)

def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)

def _date_parts(raw: pd.Series) -> pd.Series:
    values = _text(raw).str.replace(r"\.0$", "", regex=True)
    compact = values.str.replace(r"[年月日/\.\-\s]", "", regex=True)
    is_roc = compact.str.fullmatch(r"\d{6,7}", na=False)
    roc_year = (pd.to_numeric(compact.str[:-4], errors="coerce") + 1911).astype("Int64").astype("string")
    compact = compact.where(~is_roc, roc_year + compact.str[-4:])
    pieces = values.str.extract(r"^(\d{2,4})[年/\.\-](\d{1,2})[月/\.\-](\d{1,2})日?$")
    year = pd.to_numeric(pieces[0], errors="coerce")
    year = year.where(year >= 1911, year + 1911)
    separated = year.astype("Int64").astype("string") + pieces[1].str.zfill(2) + pieces[2].str.zfill(2)
    compact = compact.where(pieces[0].isna(), separated)
    return pd.to_datetime(compact, format="%Y%m%d", errors="coerce")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a fixed schema without reading prices or identifiers.

    Excludes 編號, 總價元, 建築完成年月 (unusable in provided data),
    有無管理組織 (constant), 解約情形 (potentially post-transaction),
    and raw 備註 (may contain prices). Extreme numeric values are retained.
    Parsed transaction years outside 1990..2030 become missing with a flag;
    they are never clipped to a boundary.
    """
    out: dict[str, pd.Series] = {}
    categorical = {
        "district": ("鄉鎮市區",),
        "transaction_type": ("交易標的",),
        "urban_zone": ("都市土地使用分區",),
        "nonurban_zone": ("非都市土地使用分區",),
        "nonurban_use": ("非都市土地使用編定",),
        "building_type": ("建物型態",),
        "main_use": ("主要用途",),
        "main_material": ("主要建材",),
        "partition": ("建物現況格局-隔間", "建物現況格局－隔間"),
        "parking_type": ("車位類別",),
        "project_name": ("建案名稱",),
        "building_and_unit": ("棟及號",),
        "transfer_floor_raw": ("移轉層次",),
    }
    for name, source_names in categorical.items():
        out[name] = _cat(_source(df, *source_names))

    address = _text(_source(df, "土地位置建物門牌")).str.replace("台", "臺", regex=False)
    out["address_raw"] = _cat(address)
    city = address.str.extract(r"^(.{2,3}?[縣市])", expand=False)
    out["address_city"] = _cat(city)
    after_city = address.str.replace(r"^.{2,3}?[縣市]", "", regex=True)
    district = after_city.str.extract(r"^(.{1,4}?[區鄉鎮市])", expand=False)
    out["address_district"] = _cat(district.fillna(_text(_source(df, "鄉鎮市區"))))
    after_district = after_city.str.replace(r"^.{1,4}?[區鄉鎮市]", "", regex=True)
    road = after_district.str.extract(r"^(.{1,18}?(?:大道|路|街))", expand=False)
    out["address_road"] = _cat(road)
    out["address_section"] = _cat(after_district.str.extract(r"([0-9" + _CN + r"]+段)", expand=False))
    out["address_locality"] = _cat(after_district.str.extract(r"^(.{1,12}?(?:段|里|村))", expand=False))
    out["address_is_land_parcel"] = address.str.contains(r"地號|段.*小段", na=False).astype("float32")
    out["location_road"] = _cat(city.fillna("") + "|" + district.fillna(out["district"]) + "|" + road.fillna(MISSING))

    unit = _text(_source(df, "棟及號"))
    out["building_block"] = _cat(unit.str.extract(r"^(.{1,12}?[棟幢])", expand=False))
    unit_floor = unit.str.extract(r"((?:地下|地上|B)?[0-9" + _CN + r"]+)[樓層Ff]", expand=False)
    out["unit_floor"] = unit_floor.fillna(MISSING).map(lambda x: _floor_tuple(x)[2]).astype("float32")

    numeric = {
        "land_area": ("土地移轉總面積平方公尺",),
        "building_area": ("建物移轉總面積平方公尺",),
        "parking_area": ("車位移轉總面積平方公尺",),
        "rooms": ("建物現況格局-房", "建物現況格局－房"),
        "living_rooms": ("建物現況格局-廳", "建物現況格局－廳"),
        "bathrooms": ("建物現況格局-衛", "建物現況格局－衛"),
    }
    for name, source_names in numeric.items():
        number = _numeric(_source(df, *source_names))
        out[name] = number
        out[name + "_log1p"] = np.log1p(number.where(number >= 0))
        out[name + "_negative"] = (number < 0).astype("float32")
    for area in ("land_area", "building_area", "parking_area"):
        out[area + "_zero"] = (out[area] == 0).astype("float32")
    out["large_area"] = ((out["land_area"] > 10000) | (out["building_area"] > 2000)).astype("float32")
    out["unusual_layout"] = ((out["rooms"] > 20) | (out["living_rooms"] > 10) | (out["bathrooms"] > 20)).astype("float32")

    total_raw = _text(_source(df, "總樓層數"))
    total_numeric = _numeric(_source(df, "總樓層數"))
    total_parsed = total_raw.fillna(MISSING).map(lambda x: _floor_tuple(x)[1])
    out["total_floors"] = total_numeric.fillna(total_parsed).astype("float32")
    floor_values = out["transfer_floor_raw"].map(_floor_tuple)
    floor_names = ("floor_min", "floor_max", "floor_mean", "floor_count", "floor_span",
                   "is_basement", "is_entire_building", "is_rooftop")
    for i, name in enumerate(floor_names):
        out[name] = floor_values.map(lambda value, idx=i: value[idx]).astype("float32")
    out["relative_floor"] = _safe_ratio(out["floor_mean"], out["total_floors"])
    out["floors_below_top"] = out["total_floors"] - out["floor_max"]
    out["is_top_floor"] = ((out["floor_max"] == out["total_floors"]) & (out["total_floors"] > 0)).astype("float32")

    counts = _text(_source(df, "交易筆棟數"))
    for label, name in (("土地", "land_count"), ("建物", "building_count"), ("車位", "parking_count")):
        value = counts.str.extract(label + r"\s*([0-9" + _CN + r"]+)", expand=False)
        out[name] = value.fillna("").map(_chinese_number).astype("float32")
    out["has_parking"] = ((out["parking_count"] > 0) | (out["parking_area"] > 0) |
                          (~out["parking_type"].isin([MISSING, "無", "無車位"]))).astype("float32")
    out["indoor_area"] = out["building_area"] - out["parking_area"].fillna(0)
    out["parking_fraction"] = _safe_ratio(out["parking_area"], out["building_area"])
    out["land_building_ratio"] = _safe_ratio(out["land_area"], out["building_area"])
    out["area_per_room"] = _safe_ratio(out["building_area"], out["rooms"])
    out["area_per_bathroom"] = _safe_ratio(out["building_area"], out["bathrooms"])
    out["bathrooms_per_room"] = _safe_ratio(out["bathrooms"], out["rooms"])
    out["area_per_building"] = _safe_ratio(out["building_area"], out["building_count"])

    date = _date_parts(_source(df, "交易年月日"))
    outside = date.notna() & ~date.dt.year.between(1990, 2030)
    out["trade_date_out_of_range"] = outside.astype("float32")
    date = date.mask(outside)
    out["trade_year"] = date.dt.year
    out["trade_month"] = date.dt.month
    out["trade_quarter"] = date.dt.quarter
    out["trade_day"] = date.dt.day
    out["trade_month_index"] = (date.dt.year - 2000) * 12 + date.dt.month
    out["trade_day_index"] = (date - pd.Timestamp("2000-01-01")).dt.days
    out["trade_month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
    out["trade_month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)
    out["trade_date_invalid"] = date.isna().astype("float32")

    remarks = _text(_source(df, "備註"))
    remark_patterns = {
        "remark_related_party": r"親友|親屬|二親等|關係人|特殊關係",
        "remark_decoration": r"裝潢|裝修|傢俱|家具|家電",
        "remark_unfinished": r"毛胚|未裝潢",
        "remark_extension": r"增建|加蓋|違建",
        "remark_presale": r"預售|預定買賣",
        "remark_partial_share": r"持分|部分產權|共有|應有部分",
        "remark_special_sale": r"急售|法拍|拍賣|瑕疵|凶宅|海砂|輻射",
        "remark_land_only": r"無建物|未辦保存|未登記建物",
    }
    for name, pattern in remark_patterns.items():
        out[name] = remarks.str.contains(pattern, na=False).astype("float32")

    result = pd.DataFrame(out, index=df.index)
    for name in result:
        if pd.api.types.is_numeric_dtype(result[name]):
            result[name] = pd.to_numeric(result[name], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
        else:
            result[name] = result[name].fillna(MISSING).astype(object)
    return result

def get_categorical_columns(X: pd.DataFrame) -> list[str]:
    """Return categorical feature names for categorical-aware estimators."""
    return [name for name in X.columns if not pd.api.types.is_numeric_dtype(X[name])]

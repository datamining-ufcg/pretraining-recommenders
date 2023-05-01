from tqdm import tqdm

from .preprocessing import to_raw_title
from .string_matching import match


def merge_dfs(df1, df2, col="title"):
    merged = df1.merge(df2, left_on=col, right_on=col)
    records = merged[["movieId_x", "movieId_y"]].to_dict("records")
    mapping = {r["movieId_x"]: r["movieId_y"] for r in records}

    item_missing = df1[~df1["movieId"].isin(merged["movieId_x"])].reset_index(drop=True)
    other_missing = df2[~df2["movieId"].isin(merged["movieId_y"])].reset_index(
        drop=True
    )

    return mapping, item_missing, other_missing


def match_dataframes(df1, df2, previous_mapping={}):
    mapped1 = list(previous_mapping.keys())
    df1 = df1[~df1["movieId"].isin(mapped1)].reset_index(drop=True)

    mapped2 = list(previous_mapping.values())
    df2 = df2[~df2["movieId"].isin(mapped2)].reset_index(drop=True)

    mapping, item_missing, other_missing = merge_dfs(df1, df2)
    mapping = {**previous_mapping, **mapping}

    item_missing["title_raw"] = to_raw_title(item_missing["title"].tolist())
    other_missing["title_raw"] = to_raw_title(other_missing["title"].tolist())

    mapping2, item_missing, other_missing = merge_dfs(
        item_missing, other_missing, col="title_raw"
    )

    mapping = {**mapping, **mapping2}

    item_raw = to_raw_title(item_missing["title"].tolist(), year=True)
    other_raw = to_raw_title(other_missing["title"].tolist(), year=True)

    for idx, i in enumerate(tqdm(item_raw)):
        bi, bv = match(i, other_raw, weights=[3, 1, 0, 0, 1])
        key = item_missing.loc[idx]["movieId"]
        if round(bv, 4) > 0.75 and (
            (key not in mapping) or (key in mapping and mapping[key][1] < bv)
        ):
            mapping[key] = (other_missing.loc[bi]["movieId"], bv)

    for k in mapping.keys():
        if isinstance(mapping[k], tuple):
            mapping[k] = mapping[k][0]

    return mapping

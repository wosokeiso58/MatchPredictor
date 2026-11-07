import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)

def build_match_table(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        raw = df["Date"]
        if raw.str.contains("-").all():
            df["Date"] = pd.to_datetime(raw, format="%Y-%m-%d")
        else:
            df["Date"] = pd.to_datetime(raw, format="%d/%m/%Y")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True).copy()
    df["MatchID"] = df.index

    odds = df[["MatchID", "B365H", "B365D", "B365A"]]

    home = df[["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]].copy()

    home = home.rename(columns={
        "HomeTeam": "Team",
        "AwayTeam": "Opponent",
        "FTHG": "GoalsScored",
        "FTAG": "GoalsConceded"
    })

    home["Points"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})
    home["IsHome"] = True
    home["MatchID"] = home.index

    away = df[["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]].copy()

    away = away.rename(columns={
        "AwayTeam": "Team",
        "HomeTeam": "Opponent",
        "FTAG": "GoalsScored",
        "FTHG": "GoalsConceded"
    })

    away["Points"] = away["FTR"].map({"A": 3, "D": 1, "H": 0})
    away["IsHome"] = False
    away["MatchID"] = away.index

    matches = pd.concat([home, away], ignore_index=True)

    matches = matches.sort_values("Date").reset_index(drop=True)
    matches.groupby("Team")
    matches["PreviousPoints"] = (matches.groupby("Team")["Points"].shift(1))

    matches["FormPoints"] = (
        matches.groupby("Team")["Points"]
        .transform(lambda x: x.shift(1).rolling(5).sum())
    )
    matches["AvgGoalsScored"] = (
        matches.groupby("Team")["GoalsScored"]
        .transform(lambda x: x.shift(1).rolling(5).mean())
    )
    matches["AvgGoalsConceded"] = (
        matches.groupby("Team")["GoalsConceded"]
        .transform(lambda x: x.shift(1).rolling(5).mean())
    )

    matches["LongFormPoints"] = (
        matches.groupby("Team")["Points"]
        .transform(lambda x: x.shift(1).rolling(20).sum())
    )
    matches["LongAvgGoalsScored"] = (
        matches.groupby("Team")["GoalsScored"]
        .transform(lambda x: x.shift(1).rolling(20).mean())
    )
    matches["LongAvgGoalsConceded"] = (
        matches.groupby("Team")["GoalsConceded"]
        .transform(lambda x: x.shift(1).rolling(20).mean())
    )

    home_rows = matches[matches["IsHome"]]
    away_rows = matches[~matches["IsHome"]]

    merged = home_rows.merge(away_rows, on="MatchID", suffixes=("_Home", "_Away"))
    merged = merged.merge(odds, on="MatchID")

    merged["FormDiff"] = merged["FormPoints_Home"] - merged["FormPoints_Away"]

    merged["LongFormDiff"] = merged["LongFormPoints_Home"] - merged["LongFormPoints_Away"]

    merged = merged[
        ["MatchID", "Date_Home", "Team_Home", "Team_Away", "FormPoints_Home", "FormPoints_Away", "AvgGoalsScored_Home",
         "AvgGoalsScored_Away", "AvgGoalsConceded_Home", "AvgGoalsConceded_Away", "FormDiff","LongFormDiff","LongFormPoints_Home", "LongFormPoints_Away","FTR_Home", "B365H", "B365D", "B365A"]].copy()
    merged = merged.rename(columns={
        "FTR_Home": "FTR",
        "Date_Home": "Date"
    })
    merged.dropna(subset=["FormDiff"], inplace=True)
    merged.dropna(subset=["LongFormDiff"], inplace=True)

    return merged.reset_index(drop=True)

result = build_match_table(["2021.csv","2122.csv","2223.csv","2324.csv", "2425.csv", "2526.csv"])
result = result[result["Date"] >= "2021-07-01"].copy()
result = result.sort_values("Date").reset_index(drop=True)

train = result[result["Date"] < "2026-01-01"].copy()
test = result[result["Date"] >= "2026-01-01"].copy()

features = ["FormPoints_Home", "FormPoints_Away", "AvgGoalsScored_Home", "AvgGoalsScored_Away",
            "AvgGoalsConceded_Home", "AvgGoalsConceded_Away","LongFormPoints_Home", "LongFormPoints_Away"]

X_train = train[features]
y_train = train["FTR"]
X_test = test[features]
y_test = test["FTR"]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# print(X_train.shape, X_test.shape)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
predictions = model.predict(X_test_s)

accuracy = (predictions == y_test).mean()

coefs = pd.DataFrame(model.coef_, index=model.classes_, columns=features)
print(coefs.round(3))
# print(pd.Series(predictions).value_counts())
#
# print(result.columns)
# print("rows in result:", len(result))
# print("train/test sizes:", len(train), len(test))
# print("model accuracy:", accuracy)
# print(result[["Team_Home", "Team_Away", "B365H", "B365D", "B365A"]].head())


odds_cols = ["B365H", "B365D", "B365A"]
picked = test[odds_cols].idxmin(axis=1)

test["BookiePick"] = picked.map({"B365H": "H", "B365D": "D", "B365A": "A"})

bookie_accuracy = (test["BookiePick"] == test["FTR"]).mean()
# print("bookmaker favourite accuracy:", bookie_accuracy)


taccuracy = (test["BookiePick"] == predictions).mean()
# print("bookmaker favourite accuracy:", taccuracy)
#
# print(test["BookiePick"].value_counts())
# print(result[["B365H", "B365D", "B365A"]].isna().sum())
test["ModelPick"] = predictions
dis = test[test["ModelPick"] != test["BookiePick"]]
# print("disagreements:", len(dis))
# print("bookie right:", (dis["BookiePick"] == dis["FTR"]).sum())
# print("model right:", (dis["ModelPick"] == dis["FTR"]).sum())
# print(dis["FTR"].value_counts())
#
# print(dis[["Team_Home", "Team_Away", "FormPoints_Home", "FormPoints_Away",
#            "B365H", "B365A", "ModelPick", "FTR"]].head(10))


print("test size:", len(test))
print("train size:", len(X_train))
print("always-home correct:", (test["FTR"] == "H").sum())
print("model correct:", (predictions == y_test).sum())
print("bookie correct:", (test["BookiePick"] == test["FTR"]).sum())
print(pd.Series(predictions).value_counts())
print(list(X_train.columns))
print(X_train.shape)



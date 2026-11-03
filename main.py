import pandas as pd
from sklearn.linear_model import LogisticRegression


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

    home_rows = matches[matches["IsHome"]]
    away_rows = matches[~matches["IsHome"]]

    merged = home_rows.merge(away_rows, on="MatchID", suffixes=("_Home", "_Away"))
    merged["FormDiff"] = merged["FormPoints_Home"] - merged["FormPoints_Away"]
    merged = merged[
        ["MatchID", "Date_Home", "Team_Home", "Team_Away", "FormPoints_Home", "FormPoints_Away", "AvgGoalsScored_Home",
         "AvgGoalsScored_Away", "AvgGoalsConceded_Home", "AvgGoalsConceded_Away", "FormDiff", "FTR_Home"]].copy()
    merged = merged.rename(columns={
        "FTR_Home": "FTR",
        "Date_Home": "Date"
    })
    merged.dropna(subset=["FormDiff"], inplace=True)

    return merged.reset_index(drop=True)

result = build_match_table(["2021.csv","2122.csv","2223.csv","2324.csv", "2425.csv", "2526.csv"])
result = result[result["Date"] >= "2021-07-01"].copy()
result = result.sort_values("Date").reset_index(drop=True)
print(len(result))
train = result[result["Date"] < "2026-01-01"].copy()
test = result[result["Date"] >= "2026-01-01"].copy()
print(len(test))

features = ["FormPoints_Home", "FormPoints_Away", "AvgGoalsScored_Home", "AvgGoalsScored_Away",
            "AvgGoalsConceded_Home", "AvgGoalsConceded_Away", "FormDiff"]

X_train = train[features]
y_train = train["FTR"]
X_test = test[features]
y_test = test["FTR"]

print(X_train.shape, X_test.shape)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = (predictions == y_test).mean()
print(accuracy)
print(pd.Series(predictions).value_counts())





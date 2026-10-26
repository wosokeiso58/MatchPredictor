import pandas as pd
df = pd.read_csv('2526.csv')

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
merged.dropna(subset=["FormDiff"], inplace=True)
strong = merged[merged["FormDiff"] < -5]
print(strong["FTR_Away"].value_counts(normalize=True))
print(len(strong))
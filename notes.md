# Model Creation

1) Define/select variables of interest
2) Any exploratory analysis
3) Preprocessing steps
  - Numeric/categorical imputation
  - Categorical encoding
  - Any potential filtering/removal
4) Train/test data split
5) Explore model(s)
6) Select final competition model
7) Model training
8) Model validation/testing
9) Make predictions for 2025 data

## Variables to explore

Input variables for team A and team B...

- Regular season win %: overall, last 10 games, last 5 games, against each conference
- Regular season average points scored: overall, last 10 games, last 5 games
- Regular season average points fielded: overall, last 10 games, last 5 games
- Regular season average rebounds: overall, last 10 games, last 5 games
- Regular season average rebounds fielded: overall, last 10 games, last 5 games
- Regular season average offensive/defensive rebounds: overall, last 10 games, last 5 games
- Regular season average offensive/defensive rebounds fielded: overall, last 10 games, last 5 games
- Regular season average turnovers: overall, last 10 games, last 5 games
- Regular season average turnovers forced: overall, last 10 games, last 5 games
- Regular season average FG%: overall, last 10 games, last 5 games
- Regular season average 3P%: overall, last 10 games, last 5 games
- Regular season average FG% fielded: overall, last 10 games, last 5 games
- Regular season average 3P% fielded: overall, last 10 games, last 5 games
- Regular season average FT%: overall, last 10 games, last 5 games
- Game location (home/away/neutral)
- Current rank: AP, KenPom
- Highest season rank: AP, KenPom
- Team coach
- Prior year postseason tourney status
- Days rest since last game
- Win streak
- Lose streak

Target variable...

- Logistic -> Binary: team A win %
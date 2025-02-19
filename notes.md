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

- Win %: overall, last 10 games, last 5 games, against each conference
- Average points scored: overall, last 10 games, last 5 games
- Average points fielded: overall, last 10 games, last 5 games
- Average rebounds: overall, last 10 games, last 5 games
- Average rebounds fielded: overall, last 10 games, last 5 games
- Average turnovers: overall, last 10 games, last 5 games
- Average turnovers forced: overall, last 10 games, last 5 games
- Average FG%: overall, last 10 games, last 5 games
- Average 3P%: overall, last 10 games, last 5 games
- Average FG% fielded: overall, last 10 games, last 5 games
- Average 3P% fielded: overall, last 10 games, last 5 games
- Average FT%: overall, last 10 games, last 5 games
- Game location (0 = "home", 1 = "away", 2 = "neutral". All will be "neutral" for final 2025 predictions)
    > When team A is "home", team B must then be "away".  
    > When team A is "neutral", team B must also be "neutral".  
- Game type (0 = "regular season", 1 = "conference tourney", 2 = "postseason". All will be "postseason" for final 2025 predictions)
    > When team A is "regular season", team B must also be "regular season".  
    > When team A is "conference tourney", team B must also be "conference tourney".  
    > When team A is "postseason", team B must also be "postseason".  
- Current rank: AP, KenPom
- Highest season rank: AP, KenPom
- Team coach
- Prior year tourney status (0 = made tourney X prior season, 1 = made tourney Y prior season, ..., N = did not make postseason prior season)
- Days of rest between most recent game

Target variable...

- Logistic -> Binary: team A win %

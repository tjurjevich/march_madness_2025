from dash import Dash, Input, Output, State
from dash import dcc, html
import polars as pl
import joblib
import numpy as np
import time 
import warnings
#from sklearn.utils.validation import UserWarning
#warnings.filterwarnings('ignore', category = UserWarning)


app = Dash(__name__)

teamNames = pl.read_csv('data/MTeams.csv')
team_name_list = teamNames.select('TeamName').to_numpy().flatten().tolist()

data_2025 = pl.read_parquet('2025_data.parquet')
model = joblib.load('ensemble_model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

# Define helper function that takes in TeamName and gets the following...
def data_pull(teamID: int, data = data_2025) -> dict:

    # Setup rolling average function for game level rolling average statistics
    cum_mean = lambda x: pl.col(x).cum_sum().truediv(pl.col(x).cum_count())

    # Select & calculate model variables. Impute values using literal values/backward fill/forward fill
    teamData = data.filter(pl.col("TeamID")==teamID).sort(["Season","TeamID","DayNum"]).with_columns(
        # Active AP/POM
        pl.col("ActiveAPRank").fill_null(value=9999),
        pl.col("ActivePOMRank").forward_fill().over(["Season","TeamID"]).fill_null(value=9999),
        pl.col("ActiveNETRank").forward_fill().over(["Season","TeamID"]).fill_null(value=9999),
        # NCAA tourney seed
        pl.col("NCAATourneySeed").fill_null(9999),
        # Fill all missing ActiveTourneyWins_School/ActiveTourneyWins_Coach with 0 (already backfilled)
        pl.col("ActiveTourneyWins_School").fill_null(0),
        pl.col("ActiveTourneyWins_Coach").fill_null(0),
        # Season best rankings (AP/POM)
        pl.col("ActiveAPRank").cum_min().forward_fill().over(["Season","TeamID"]).alias("SeasonBestAPRank").fill_null(value=9999),
        pl.col("ActivePOMRank").cum_min().forward_fill().over(["Season","TeamID"]).alias("SeasonBestPOMRank").fill_null(value=9999),
        # Win pct
        ((pl.col("WinFlag").cum_sum().over(['Season','TeamID']))/((pl.col("WinFlag").cum_sum().over(['Season','TeamID']))+(pl.col("LoseFlag").cum_sum().over(['Season','TeamID'])))).alias("RollingWinPct_Overall"),
        ((pl.col("WinFlag").rolling_sum(window_size=5).over(['Season','TeamID']))/((pl.col("WinFlag").rolling_sum(window_size=5).over(['Season','TeamID']))+(pl.col("LoseFlag").rolling_sum(window_size=5).over(['Season','TeamID'])))).alias("RollingWinPct_Last5"),
        # Power6 win% 
        ((pl.col("Power5WinFlag").cum_sum().over(['Season','TeamID']))/((pl.col("Power5WinFlag").cum_sum().over(['Season','TeamID']))+(pl.col("Power5LoseFlag").cum_sum().over(['Season','TeamID'])))).alias("RollingP5WinPct_Overall"),
        # Non-power6 win%
        ((pl.col("NonPower5WinFlag").cum_sum().over(['Season','TeamID']))/((pl.col("NonPower5WinFlag").cum_sum().over(['Season','TeamID']))+(pl.col("NonPower5LoseFlag").cum_sum().over(['Season','TeamID'])))).alias("RollingNP5WinPct_Overall"),
        # Points scored
        (cum_mean("PointsScored").over(['Season','TeamID']).alias('PointsScoredAvg_Overall')),
        (pl.col("PointsScored").rolling_mean(window_size=5).over(['Season','TeamID']).alias('PointsScoredAvg_Last5')),
        # Points allowed
        (cum_mean("PointsAllowed").over(['Season','TeamID']).alias('PointsAllowedAvg_Overall')),
        (pl.col("PointsAllowed").rolling_mean(window_size=5).over(['Season','TeamID']).alias('PointsAllowedAvg_Last5')),
        # Rebounds
        (cum_mean("TotalRebounds").over(['Season','TeamID']).alias('ReboundsAvg_Overall')),
        (pl.col("TotalRebounds").rolling_mean(window_size=5).over(['Season','TeamID']).alias('ReboundsAvg_Last5')),
        # Rebounds allowed
        (cum_mean("OppTotalRebounds").over(['Season','TeamID']).alias('OppReboundsAvg_Overall')),
        (pl.col("OppTotalRebounds").rolling_mean(window_size=5).over(['Season','TeamID']).alias('OppReboundsAvg_Last5')),
        # Offensive rebounds
        (cum_mean("OffensiveRebounds").over(['Season','TeamID']).alias('OffensiveReboundsAvg_Overall')),
        (pl.col("OffensiveRebounds").rolling_mean(window_size=5).over(['Season','TeamID']).alias('OffensiveReboundsAvg_Last5')),
        # Defensive rebounds
        (cum_mean("DefensiveRebounds").over(['Season','TeamID']).alias('DefensiveReboundsAvg_Overall')),
        (pl.col("DefensiveRebounds").rolling_mean(window_size=5).over(['Season','TeamID']).alias('DefensiveReboundsAvg_Last5')),
        # Offensive rebounds allowed
        (cum_mean("OppOffensiveRebounds").over(['Season','TeamID']).alias('OppOffensiveReboundsAvg_Overall')),
        (pl.col("OppOffensiveRebounds").rolling_mean(window_size=5).over(['Season','TeamID']).alias('OppOffensiveReboundsAvg_Last5')),
        # Defensive rebounds allowed
        (cum_mean("OppDefensiveRebounds").over(['Season','TeamID']).alias('OppDefensiveReboundsAvg_Overall')),
        (pl.col("OppDefensiveRebounds").rolling_mean(window_size=5).over(['Season','TeamID']).alias('OppDefensiveReboundsAvg_Last5')),
        # Free throw % 
        (cum_mean("FreeThrowPct").over(['Season','TeamID']).alias('FreeThrowPct_Overall')),
        (pl.col("FreeThrowPct").rolling_mean(window_size=5).over(['Season','TeamID']).alias('FreeThrowPct_Last5')),
        # Free throw % opponent
        (cum_mean("OppFreeThrowPct").over(['Season','TeamID']).alias('OppFreeThrowPct_Overall')),
        (pl.col("OppFreeThrowPct").rolling_mean(window_size=5).over(['Season','TeamID']).alias('OppFreeThrowPct_Last5')),
        # Field goal %
        (cum_mean("FieldGoalPct").over(['Season','TeamID']).alias('FieldGoalPct_Overall')),
        (pl.col('FieldGoalPct').rolling_mean(window_size=5).over(['Season','TeamID']).alias('FieldGoalPct_Last5')),
        # Field goal % allowed
        (cum_mean("OppFieldGoalPct").over(['Season','TeamID']).alias('OppFieldGoalPct_Overall')),
        (pl.col('OppFieldGoalPct').rolling_mean(window_size=5).over(['Season','TeamID']).alias('OppFieldGoalPct_Last5')),
        # 3P%
        (cum_mean("ThreePtPct").over(['Season','TeamID']).alias('ThreePtPct_Overall')),
        (pl.col('ThreePtPct').rolling_mean(window_size=5).over(['Season','TeamID']).alias('ThreePtPct_Last5')),
        # 3P% allowed
        (cum_mean("OppThreePtPct").over(['Season','TeamID']).alias('OppThreePtPct_Overall')),
        (pl.col('OppThreePtPct').rolling_mean(window_size=5).over(['Season','TeamID']).alias('OppThreePtPct_Last5'))
    ).select(["WinFlag","Season","GameID","DayNum","TeamID","RollingWinPct_Overall","RollingWinPct_Last5","GameLocation","TeamConf","RegSeasonFlag","ConfTourneyFlag","NCAATourneyFlag",
            "RollingP5WinPct_Overall","RollingNP5WinPct_Overall",
            "CoachName","ActiveTourneyWins_School","ActiveTourneyWins_Coach","NCAATourneySeed","ActiveAPRank","ActivePOMRank","ActiveNETRank","SeasonBestAPRank","SeasonBestPOMRank",
            "PointsScoredAvg_Overall","PointsScoredAvg_Last5","PointsAllowedAvg_Overall","PointsAllowedAvg_Last5",
            "ReboundsAvg_Overall","ReboundsAvg_Last5","OppReboundsAvg_Overall","OppReboundsAvg_Last5",
            "FreeThrowPct_Overall","FreeThrowPct_Last5","OppFreeThrowPct_Overall","OppFreeThrowPct_Last5",
            "FieldGoalPct_Overall","FieldGoalPct_Last5","OppFieldGoalPct_Overall","OppFieldGoalPct_Last5",
            "ThreePtPct_Overall","ThreePtPct_Last5","OppThreePtPct_Overall","OppThreePtPct_Last5"])

    # Return averages/numbers from most recent game
    teamData = teamData.with_columns(
        pl.col("DayNum").rank(method="dense", descending = True).over(["TeamID"]).alias("GameNum")
    ).filter(pl.col("GameNum") == 1).fill_nan(0).fill_null(0)

    return teamData.to_dict(as_series=False)
    



app.layout = html.Div([
    html.H1("March Madness Mania: 2025 Predictor", id = 'main-header'),

    # Teams div
    html.Div([
        # Team 1 info
        html.Div([

            html.Div([
                html.Label("Team 1", id = 'team-1-name-label'),
                dcc.Dropdown(team_name_list, id = 'team-1-name')
            ], className = 'team-name-div'),
            
            html.Div([
                html.Label("Team 1 Tourney Seed", id = "team-1-seed-label"),
                dcc.Input(id='team-1-seed', type='number', min=1, max=16, step=1)
            ], className = 'team-seed-div')

        ], className = 'team-1-container'),
        # Team 2 info
                html.Div([

            html.Div([
                html.Label("Team 2", id = 'team-2-name-label'),
                dcc.Dropdown(team_name_list, id = 'team-2-name')
            ], className = 'team-name-div'),
            
            html.Div([
                html.Label("Team 2 Tourney Seed", id = "team-2-seed-label"),
                dcc.Input(id='team-2-seed', type='number', min=1, max=16, step=1)
            ], className = 'team-seed-div')

        ], className = 'team-2-container')
    ]),

    # Results div
    html.Div([
    html.Div(
        [
            html.Label("ML Model Prediction...", id = 'result-label'),
            html.Div(id = 'result')
        ],
        className = 'result-div'
    )],
    className = 'result-container'
    )
]
)


@app.callback(
    Output('result', 'children'),
    [Input('team-1-name','value'), Input('team-1-seed','value'),
     Input('team-2-name','value'), Input('team-2-seed','value')]
)
def predict(team_1_name, team_1_seed, team_2_name, team_2_seed):

    if None not in [team_1_name, team_1_seed, team_2_name, team_2_seed]:
        team_1_id = teamNames.filter(pl.col("TeamName") == team_1_name)["TeamID"][0]
        team_2_id = teamNames.filter(pl.col("TeamName") == team_2_name)["TeamID"][0]

        team_1_data = data_pull(teamID = team_1_id)
        team_2_data = data_pull(teamID = team_2_id)

        # Constant variables regardless of team/game
        game_location = "N"
        reg_season_flag, conf_tourney_flag, ncaa_tourney_flag = 0, 0, 1
        season = 2025

        # Team 1 categorical
        team_1_conf = team_1_data['TeamConf'][0]
        team_1_coach = team_1_data['CoachName'][0]
        team_1_active_tourney_wins_school = team_1_data['ActiveTourneyWins_School'][0]
        team_1_active_tourney_wins_coach = team_1_data['ActiveTourneyWins_Coach'][0]
        team_1_active_ap = team_1_data['ActiveAPRank'][0]
        team_1_active_net = team_1_data['ActiveNETRank'][0]
        team_1_active_pom = team_1_data['ActivePOMRank'][0]
        team_1_best_ap = team_1_data['SeasonBestAPRank'][0]
        team_1_best_pom = team_1_data['SeasonBestPOMRank'][0]
        #'GameLocation_A', 'TeamConf_A','TeamConf_B','CoachName_A','CoachName_B','TeamID_A','TeamID_B','Season','NCAATourneySeed_A','NCAATourneySeed_B','ActiveTourneyWins_School_A','ActiveTourneyWins_School_B','ActiveTourneyWins_Coach_A','ActiveTourneyWins_Coach_B','ActiveAPRank_A','ActivePOMRank_A','ActiveNETRank_A','SeasonBestAPRank_A','SeasonBestPOMRank_A','ActiveAPRank_B','ActivePOMRank_B','ActiveNETRank_B','SeasonBestAPRank_B','SeasonBestPOMRank_B'

        # Team 1 numeric: no standardization
        team_1_winpct_overall = team_1_data['RollingWinPct_Overall'][0]
        team_1_winpct_last5 = team_1_data['RollingWinPct_Last5'][0]
        team_1_p5_winpct = team_1_data['RollingP5WinPct_Overall'][0]
        team_1_np5_winpct = team_1_data['RollingNP5WinPct_Overall'][0]
        team_1_ftp_overall = team_1_data['FreeThrowPct_Overall'][0]
        team_1_ftp_last5 = team_1_data['FreeThrowPct_Last5'][0]
        team_1_opp_ftp_overall = team_1_data['OppFreeThrowPct_Overall'][0]
        team_1_opp_ftp_last5 = team_1_data['OppFreeThrowPct_Last5'][0]
        team_1_fgp_overall = team_1_data['FieldGoalPct_Overall'][0]
        team_1_fgp_last5 = team_1_data['FieldGoalPct_Last5'][0]
        team_1_opp_fgp_overall = team_1_data['OppFieldGoalPct_Overall'][0]
        team_1_opp_fgp_last5 = team_1_data['OppFieldGoalPct_Last5'][0]
        team_1_tpp_overall = team_1_data['ThreePtPct_Overall'][0]
        team_1_tpp_last5 = team_1_data['ThreePtPct_Last5'][0]
        team_1_opp_tpp_overall = team_1_data['OppThreePtPct_Overall'][0]
        team_1_opp_tpp_last5 = team_1_data['OppThreePtPct_Last5'][0]
        # "RollingWinPct_Overall_A","RollingWinPct_Last5_A","RollingP5WinPct_Overall_A","RollingNP5WinPct_Overall_A","FreeThrowPct_Overall_A","FreeThrowPct_Last5_A","OppFreeThrowPct_Overall_A","OppFreeThrowPct_Last5_A","FieldGoalPct_Overall_A","FieldGoalPct_Last5_A","OppFieldGoalPct_Overall_A","OppFieldGoalPct_Last5_A","ThreePtPct_Overall_A","ThreePtPct_Last5_A","OppThreePtPct_Overall_A","OppThreePtPct_Last5_A"

        # Team 1 numeric: standardization
        team_1_scoring_avg_overall = team_1_data['PointsScoredAvg_Overall'][0]
        team_1_scoring_avg_last5 = team_1_data['PointsScoredAvg_Last5'][0]
        team_1_scoring_allowed_overall = team_1_data['PointsAllowedAvg_Overall'][0]
        team_1_scoring_allowed_last5 = team_1_data['PointsAllowedAvg_Last5'][0]
        team_1_rebound_avg_overall = team_1_data['ReboundsAvg_Overall'][0]
        team_1_rebound_avg_last5 = team_1_data['ReboundsAvg_Last5'][0]
        team_1_opp_rebound_avg_overall = team_1_data['OppReboundsAvg_Overall'][0]
        team_1_opp_rebound_avg_last5 = team_1_data['OppReboundsAvg_Last5'][0]


        # Team 2 categorical
        team_2_conf = team_2_data['TeamConf'][0]
        team_2_coach = team_2_data['CoachName'][0]
        team_2_active_tourney_wins_school = team_2_data['ActiveTourneyWins_School'][0]
        team_2_active_tourney_wins_coach = team_2_data['ActiveTourneyWins_Coach'][0]
        team_2_active_ap = team_2_data['ActiveAPRank'][0]
        team_2_active_net = team_2_data['ActiveNETRank'][0]
        team_2_active_pom = team_2_data['ActivePOMRank'][0]
        team_2_best_ap = team_2_data['SeasonBestAPRank'][0]
        team_2_best_pom = team_2_data['SeasonBestPOMRank'][0]

        # Team 2 numeric: no standardization
        team_2_winpct_overall = team_2_data['RollingWinPct_Overall'][0]
        team_2_winpct_last5 = team_2_data['RollingWinPct_Last5'][0]
        team_2_p5_winpct = team_2_data['RollingP5WinPct_Overall'][0]
        team_2_np5_winpct = team_2_data['RollingNP5WinPct_Overall'][0]
        team_2_ftp_overall = team_2_data['FreeThrowPct_Overall'][0]
        team_2_ftp_last5 = team_2_data['FreeThrowPct_Last5'][0]
        team_2_opp_ftp_overall = team_2_data['OppFreeThrowPct_Overall'][0]
        team_2_opp_ftp_last5 = team_2_data['OppFreeThrowPct_Last5'][0]
        team_2_fgp_overall = team_2_data['FieldGoalPct_Overall'][0]
        team_2_fgp_last5 = team_2_data['FieldGoalPct_Last5'][0]
        team_2_opp_fgp_overall = team_2_data['OppFieldGoalPct_Overall'][0]
        team_2_opp_fgp_last5 = team_2_data['OppFieldGoalPct_Last5'][0]
        team_2_tpp_overall = team_2_data['ThreePtPct_Overall'][0]
        team_2_tpp_last5 = team_2_data['ThreePtPct_Last5'][0]
        team_2_opp_tpp_overall = team_2_data['OppThreePtPct_Overall'][0]
        team_2_opp_tpp_last5 = team_2_data['OppThreePtPct_Last5'][0]


        # Team 2 numeric: standardization
        team_2_scoring_avg_overall = team_2_data['PointsScoredAvg_Overall'][0]
        team_2_scoring_avg_last5 = team_2_data['PointsScoredAvg_Last5'][0]
        team_2_scoring_allowed_overall = team_2_data['PointsAllowedAvg_Overall'][0]
        team_2_scoring_allowed_last5 = team_2_data['PointsAllowedAvg_Last5'][0]
        team_2_rebound_avg_overall = team_2_data['ReboundsAvg_Overall'][0]
        team_2_rebound_avg_last5 = team_2_data['ReboundsAvg_Last5'][0]
        team_2_opp_rebound_avg_overall = team_2_data['OppReboundsAvg_Overall'][0]
        team_2_opp_rebound_avg_last5 = team_2_data['OppReboundsAvg_Last5'][0]


        # Combine two teams categorical data, then transform.
        cat_data = np.array([[game_location, team_1_conf, team_2_conf, team_1_coach, team_2_coach, team_1_id, team_2_id, season, team_1_seed, team_2_seed, team_1_active_tourney_wins_school, team_2_active_tourney_wins_school, team_1_active_tourney_wins_coach, team_2_active_tourney_wins_coach, team_1_active_ap, team_1_active_pom, team_1_active_net, team_1_best_ap, team_1_best_pom, team_2_active_ap, team_2_active_pom, team_2_active_net, team_2_best_ap, team_2_best_pom]])
        transformed_cat_data = encoder.transform(cat_data)
        final_cat_data = np.hstack((transformed_cat_data, np.array([[reg_season_flag, conf_tourney_flag, ncaa_tourney_flag]])))

        # Combine two teams numeric data, then normalize.
        """
        #'PointsScoredAvg_Overall_A', 'PointsScoredAvg_Last5_A', 'PointsAllowedAvg_Overall_A', 'PointsAllowedAvg_Last5_A', 'ReboundsAvg_Overall_A', 'ReboundsAvg_Last5_A', 'OppReboundsAvg_Overall_A', 'OppReboundsAvg_Last5_A', 'PointsScoredAvg_Overall_B', 'PointsScoredAvg_Last5_B', 'PointsAllowedAvg_Overall_B', 'PointsAllowedAvg_Last5_B', 'ReboundsAvg_Overall_B', 'ReboundsAvg_Last5_B', 'OppReboundsAvg_Overall_B', 'OppReboundsAvg_Last5_B
        """
        numeric_data_1 = np.array([[team_1_scoring_avg_overall, team_1_scoring_avg_last5, team_1_scoring_allowed_overall, team_1_scoring_allowed_last5, team_1_rebound_avg_overall, team_1_rebound_avg_last5, team_1_opp_rebound_avg_overall, team_1_opp_rebound_avg_last5, team_2_scoring_avg_overall, team_2_scoring_avg_last5, team_2_scoring_allowed_overall, team_2_scoring_allowed_last5, team_2_rebound_avg_overall, team_2_rebound_avg_last5, team_2_opp_rebound_avg_overall, team_2_opp_rebound_avg_last5]])
        numeric_data_2 = np.array([[team_1_winpct_overall, team_1_winpct_last5, team_1_p5_winpct, team_1_np5_winpct, team_1_ftp_overall, team_1_ftp_last5, team_1_opp_ftp_overall, team_1_opp_ftp_last5, team_1_fgp_overall, team_1_fgp_last5, team_1_opp_fgp_overall, team_1_opp_fgp_last5, team_1_tpp_overall, team_1_tpp_last5, team_1_opp_tpp_overall, team_1_opp_tpp_last5, team_2_winpct_overall, team_2_winpct_last5, team_2_p5_winpct, team_2_np5_winpct, team_2_ftp_overall, team_2_ftp_last5, team_2_opp_ftp_overall, team_2_opp_ftp_last5, team_2_fgp_overall, team_2_fgp_last5, team_2_opp_fgp_overall, team_2_opp_fgp_last5, team_2_tpp_overall, team_2_tpp_last5, team_2_opp_tpp_overall, team_2_opp_tpp_last5]])
        scaled_numeric_data = scaler.transform(numeric_data_1)
        final_numeric_data = np.hstack((scaled_numeric_data, numeric_data_2))
        
        prediction_features = np.hstack((final_cat_data, final_numeric_data))
        if model.predict(prediction_features) == 1:
            return_string = team_1_name
        else:
            return_string = team_2_name

        time.sleep(2)

        return html.Div(return_string)




if __name__ == '__main__':
    app.run_server(debug=True)

import polars as pl 
import numpy as np

# Load regular season stats
regSeason = pl.read_csv('data/MRegularSeasonDetailedResults.csv')
regSeason = regSeason.with_columns(
    pl.lit(1).alias("RegSeasonFlag"),
    pl.lit(0).alias("NCAATourneyFlag")
)

# Load postseason stats
postSeason = pl.read_csv('data/MNCAATourneyDetailedResults.csv')
postSeason = postSeason.with_columns(
    pl.lit(0).alias("RegSeasonFlag"),
    pl.lit(1).alias("NCAATourneyFlag")
)

# Load compact postseason results
postSeasonCompact = pl.read_csv('data/MNCAATourneyCompactResults.csv')

# Vstack/concat regSeason & postSeason to get stats for entirety of season
entireSeason = pl.concat([regSeason, postSeason], how = 'vertical')
entireSeason = entireSeason.with_columns(
    pl.arange(0,entireSeason.height).alias("GameID")
)

# Load conference tourney matchups/results
cTourney = pl.read_csv('data/MConferenceTourneyGames.csv')

# Load team conference association data
mConferences = pl.read_csv('data/MTeamConferences.csv')

# Load coaching data
mCoaches = pl.read_csv('data/MTeamCoaches.csv')

# Load ranking data
apRankings = pl.read_csv('data/rankings/MMasseyOrdinals_AP.csv')
pomRankings = pl.read_csv('data/rankings/MMasseyOrdinals_POM.csv')
netRankings = pl.read_csv('data/rankings/MMasseyOrdinals_NET.csv')

# Load seeding data
tourneySeeds = pl.read_csv('data/MNCAATourneySeeds.csv')

# Create dictionary for NET Quads
net_quad_criteria = {
    'quad1':{
        'H':{'min':1,'max':30},
        'N':{'min':1,'max':50},
        'A':{'min':1,'max':75}
    },
    'quad2':{
        'H':{'min':31,'max':75},
        'N':{'min':51,'max':100},
        'A':{'min':76,'max':135}
    },
    'quad3':{
        'H':{'min':76,'max':160},
        'N':{'min':101,'max':200},
        'A':{'min':136,'max':240}
    },
    'quad4':{
        'H':{'min':161},
        'N':{'min':201},
        'A':{'min':241}
    }
}

# Set conference tourney flag
entireSeason = entireSeason.join(cTourney, how='left', on = ['Season', 'DayNum','WTeamID','LTeamID']).with_columns(
    pl.when(pl.col('ConfAbbrev').is_not_null())
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .alias("ConfTourneyFlag")
)
entireSeason = entireSeason.drop("ConfAbbrev")

# Join to team conference data
entireSeason = entireSeason.join(other = mConferences, how='left', left_on = ['Season','WTeamID'], right_on = ['Season','TeamID']).rename({"ConfAbbrev":"WConf"})\
        .join(mConferences, how='left', left_on = ['Season','LTeamID'], right_on = ['Season','TeamID']).rename({"ConfAbbrev":"LConf"})


"""
power6 = acc, big_east, big_ten, pac_ten, pac_twelve, big_twelve, sec
non power6 = all others
"""

min_teamID = min(entireSeason.select(pl.col("WTeamID").min()).to_numpy()[0][0], entireSeason.select(pl.col("WTeamID").min()).to_numpy()[0][0])
max_teamID = max(entireSeason.select(pl.col("WTeamID").max()).to_numpy()[0][0], entireSeason.select(pl.col("WTeamID").max()).to_numpy()[0][0])

for teamID in range(min_teamID, max_teamID+1):
    if teamID == min_teamID:
        winningData = entireSeason.filter(pl.col("WTeamID") == teamID)\
                        .select(["Season","GameID","RegSeasonFlag","ConfTourneyFlag", "NCAATourneyFlag","WConf","LConf","WTeamID","DayNum","WScore","WLoc","WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl","WBlk","WPF","LScore","LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst","LTO","LStl","LBlk","LPF"])
        winningData = winningData.with_columns(
            pl.lit(1).alias("WinFlag"), pl.lit(0).alias("LoseFlag")
        )
        winningData = winningData.rename({
            "WConf":"TeamConf",
            "WTeamID":"TeamID",
            "WScore":"PointsScored",
            "WLoc":"GameLocation",
            "WFGM":"FieldGoalsMade",
            "WFGA":"FieldGoalAttempts",
            "WFGM3":"ThreePointersMade",
            "WFGA3":"ThreePointerAttempts",
            "WFTM":"FreeThrowsMade",
            "WFTA":"FreeThrowAttempts",
            "WOR":"OffensiveRebounds",
            "WDR":"DefensiveRebounds",
            "WAst":"Assists",
            "WTO":"Turnovers",
            "WStl":"Steals",
            "WBlk":"Blocks",
            "WPF":"PlayerFouls",
            "LScore":"PointsAllowed",
            "LConf":"OppConf",
            "LFGM":"OppFieldGoalsMade",
            "LFGA":"OppFieldGoalAttempts",
            "LFGM3":"OppThreePointersMade",
            "LFGA3":"OppThreePointerAttempts",
            "LFTM":"OppFreeThrowsMade",
            "LFTA":"OppFreeThrowAttempts",
            "LOR":"OppOffensiveRebounds",
            "LDR":"OppDefensiveRebounds",
            "LAst":"OppAssists",
            "LTO":"OppTurnovers",
            "LStl":"OppSteals",
            "LBlk":"OppBlocks",
            "LPF":"OppPlayerFouls"
        })
        winningData = winningData.select(["GameID","Season","RegSeasonFlag","ConfTourneyFlag", "NCAATourneyFlag","DayNum","TeamConf","OppConf","TeamID","PointsScored","GameLocation","FieldGoalsMade","FieldGoalAttempts","ThreePointersMade","ThreePointerAttempts"\
                                          ,"FreeThrowsMade","FreeThrowAttempts","OffensiveRebounds","DefensiveRebounds","Assists","Turnovers","Steals","Blocks"\
                                          ,"PlayerFouls","PointsAllowed","OppFieldGoalsMade","OppFieldGoalAttempts","OppThreePointersMade","OppThreePointerAttempts"\
                                          ,"OppFreeThrowsMade","OppFreeThrowAttempts","OppOffensiveRebounds","OppDefensiveRebounds","OppAssists","OppTurnovers"\
                                          ,"OppSteals","OppBlocks","OppPlayerFouls","WinFlag","LoseFlag"])
        #print(winningData.columns)
        
        losingData = entireSeason.filter(pl.col("LTeamID") == teamID)\
                        .select(["Season","GameID","RegSeasonFlag","ConfTourneyFlag", "NCAATourneyFlag","LConf","WConf","LTeamID","DayNum","WScore","WLoc","WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl","WBlk","WPF","LScore","LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst","LTO","LStl","LBlk","LPF"])
        losingData = losingData.with_columns(pl.lit(0).alias("WinFlag"), pl.lit(1).alias("LoseFlag"))
        losingData = losingData.with_columns(
            pl.when(pl.col("WLoc") == "H")
            .then(pl.lit("A"))
            .when(pl.col("WLoc") == "A")
            .then(pl.lit("H"))
            .otherwise(pl.lit("N"))
            .alias("LLoc")
        )
        losingData = losingData.drop(["WLoc"])
        losingData = losingData.rename({
            "WConf":"OppConf",
            "LConf":"TeamConf",
            "LTeamID":"TeamID",
            "LScore":"PointsScored",
            "LLoc":"GameLocation",
            "LFGM":"FieldGoalsMade",
            "LFGA":"FieldGoalAttempts",
            "LFGM3":"ThreePointersMade",
            "LFGA3":"ThreePointerAttempts",
            "LFTM":"FreeThrowsMade",
            "LFTA":"FreeThrowAttempts",
            "LOR":"OffensiveRebounds",
            "LDR":"DefensiveRebounds",
            "LAst":"Assists",
            "LTO":"Turnovers",
            "LStl":"Steals",
            "LBlk":"Blocks",
            "LPF":"PlayerFouls",
            "WScore":"PointsAllowed",
            "WFGM":"OppFieldGoalsMade",
            "WFGA":"OppFieldGoalAttempts",
            "WFGM3":"OppThreePointersMade",
            "WFGA3":"OppThreePointerAttempts",
            "WFTM":"OppFreeThrowsMade",
            "WFTA":"OppFreeThrowAttempts",
            "WOR":"OppOffensiveRebounds",
            "WDR":"OppDefensiveRebounds",
            "WAst":"OppAssists",
            "WTO":"OppTurnovers",
            "WStl":"OppSteals",
            "WBlk":"OppBlocks",
            "WPF":"OppPlayerFouls"})
        losingData = losingData.select(["GameID","Season","RegSeasonFlag","ConfTourneyFlag", "NCAATourneyFlag","DayNum","TeamConf","OppConf","TeamID","PointsScored","GameLocation","FieldGoalsMade","FieldGoalAttempts","ThreePointersMade","ThreePointerAttempts"\
                                    ,"FreeThrowsMade","FreeThrowAttempts","OffensiveRebounds","DefensiveRebounds","Assists","Turnovers","Steals","Blocks"\
                                    ,"PlayerFouls","PointsAllowed","OppFieldGoalsMade","OppFieldGoalAttempts","OppThreePointersMade","OppThreePointerAttempts"\
                                    ,"OppFreeThrowsMade","OppFreeThrowAttempts","OppOffensiveRebounds","OppDefensiveRebounds","OppAssists","OppTurnovers"\
                                    ,"OppSteals","OppBlocks","OppPlayerFouls","WinFlag","LoseFlag"])
        
        allData =  pl.concat([winningData, losingData], how='vertical')
    
    else:
        winningData = entireSeason.filter(pl.col("WTeamID") == teamID)\
                        .select(["Season","GameID","RegSeasonFlag","ConfTourneyFlag", "NCAATourneyFlag","WTeamID","WConf","LConf","DayNum","WScore","WLoc","WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl","WBlk","WPF","LScore","LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst","LTO","LStl","LBlk","LPF"])
        winningData = winningData.with_columns(pl.lit(1).alias("WinFlag"), pl.lit(0).alias("LoseFlag"))
        winningData = winningData.rename({
            "WConf":"TeamConf",
            "LConf":"OppConf",
            "WTeamID":"TeamID",
            "WScore":"PointsScored",
            "WLoc":"GameLocation",
            "WFGM":"FieldGoalsMade",
            "WFGA":"FieldGoalAttempts",
            "WFGM3":"ThreePointersMade",
            "WFGA3":"ThreePointerAttempts",
            "WFTM":"FreeThrowsMade",
            "WFTA":"FreeThrowAttempts",
            "WOR":"OffensiveRebounds",
            "WDR":"DefensiveRebounds",
            "WAst":"Assists",
            "WTO":"Turnovers",
            "WStl":"Steals",
            "WBlk":"Blocks",
            "WPF":"PlayerFouls",
            "LScore":"PointsAllowed",
            "LFGM":"OppFieldGoalsMade",
            "LFGA":"OppFieldGoalAttempts",
            "LFGM3":"OppThreePointersMade",
            "LFGA3":"OppThreePointerAttempts",
            "LFTM":"OppFreeThrowsMade",
            "LFTA":"OppFreeThrowAttempts",
            "LOR":"OppOffensiveRebounds",
            "LDR":"OppDefensiveRebounds",
            "LAst":"OppAssists",
            "LTO":"OppTurnovers",
            "LStl":"OppSteals",
            "LBlk":"OppBlocks",
            "LPF":"OppPlayerFouls"
        })
        winningData = winningData.select(["GameID","Season","RegSeasonFlag","ConfTourneyFlag", "NCAATourneyFlag","DayNum","TeamConf","OppConf","TeamID","PointsScored","GameLocation","FieldGoalsMade","FieldGoalAttempts","ThreePointersMade","ThreePointerAttempts"\
                                          ,"FreeThrowsMade","FreeThrowAttempts","OffensiveRebounds","DefensiveRebounds","Assists","Turnovers","Steals","Blocks"\
                                          ,"PlayerFouls","PointsAllowed","OppFieldGoalsMade","OppFieldGoalAttempts","OppThreePointersMade","OppThreePointerAttempts"\
                                          ,"OppFreeThrowsMade","OppFreeThrowAttempts","OppOffensiveRebounds","OppDefensiveRebounds","OppAssists","OppTurnovers"\
                                          ,"OppSteals","OppBlocks","OppPlayerFouls","WinFlag","LoseFlag"])
        #print(winningData.columns)
        
        losingData = entireSeason.filter(pl.col("LTeamID") == teamID)\
                        .select(["Season","GameID","RegSeasonFlag","ConfTourneyFlag", "NCAATourneyFlag","WConf","LConf","LTeamID","DayNum","WScore","WLoc","WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl","WBlk","WPF","LScore","LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst","LTO","LStl","LBlk","LPF"])
        losingData = losingData.with_columns(pl.lit(0).alias("WinFlag"), pl.lit(1).alias("LoseFlag"))
        losingData = losingData.with_columns(
            pl.when(pl.col("WLoc") == "H")
            .then(pl.lit("A"))
            .when(pl.col("WLoc") == "A")
            .then(pl.lit("H"))
            .otherwise(pl.lit("N"))
            .alias("LLoc")
        )
        losingData = losingData.drop(["WLoc"])
        losingData = losingData.rename({
            "WConf":"OppConf",
            "LConf":"TeamConf",
            "LTeamID":"TeamID",
            "LScore":"PointsScored",
            "LLoc":"GameLocation",
            "LFGM":"FieldGoalsMade",
            "LFGA":"FieldGoalAttempts",
            "LFGM3":"ThreePointersMade",
            "LFGA3":"ThreePointerAttempts",
            "LFTM":"FreeThrowsMade",
            "LFTA":"FreeThrowAttempts",
            "LOR":"OffensiveRebounds",
            "LDR":"DefensiveRebounds",
            "LAst":"Assists",
            "LTO":"Turnovers",
            "LStl":"Steals",
            "LBlk":"Blocks",
            "LPF":"PlayerFouls",
            "WScore":"PointsAllowed",
            "WFGM":"OppFieldGoalsMade",
            "WFGA":"OppFieldGoalAttempts",
            "WFGM3":"OppThreePointersMade",
            "WFGA3":"OppThreePointerAttempts",
            "WFTM":"OppFreeThrowsMade",
            "WFTA":"OppFreeThrowAttempts",
            "WOR":"OppOffensiveRebounds",
            "WDR":"OppDefensiveRebounds",
            "WAst":"OppAssists",
            "WTO":"OppTurnovers",
            "WStl":"OppSteals",
            "WBlk":"OppBlocks",
            "WPF":"OppPlayerFouls"})
        losingData = losingData.select(["GameID","Season","RegSeasonFlag","ConfTourneyFlag", "NCAATourneyFlag","DayNum","TeamConf","OppConf","TeamID","PointsScored","GameLocation","FieldGoalsMade","FieldGoalAttempts","ThreePointersMade","ThreePointerAttempts"\
                                    ,"FreeThrowsMade","FreeThrowAttempts","OffensiveRebounds","DefensiveRebounds","Assists","Turnovers","Steals","Blocks"\
                                    ,"PlayerFouls","PointsAllowed","OppFieldGoalsMade","OppFieldGoalAttempts","OppThreePointersMade","OppThreePointerAttempts"\
                                    ,"OppFreeThrowsMade","OppFreeThrowAttempts","OppOffensiveRebounds","OppDefensiveRebounds","OppAssists","OppTurnovers"\
                                    ,"OppSteals","OppBlocks","OppPlayerFouls","WinFlag","LoseFlag"])
        #print(losingData.columns)
        
        tempData =  pl.concat([winningData, losingData], how='vertical')
        allData = pl.concat([allData, tempData], how='vertical')
 

# Create new columns (total rebounds, field goal/three point/free throw %, power 5 win/loss flags)
allData = allData.with_columns(
    ((pl.col("OffensiveRebounds") + pl.col("DefensiveRebounds")).alias('TotalRebounds')),
    ((pl.col("OppOffensiveRebounds") + pl.col("OppDefensiveRebounds")).alias('OppTotalRebounds')),
    ((pl.col("FieldGoalsMade")/pl.col("FieldGoalAttempts")).alias("FieldGoalPct")),
    ((pl.col("OppFieldGoalsMade")/pl.col("OppFieldGoalAttempts")).alias("OppFieldGoalPct")),
    ((pl.col("ThreePointersMade")/pl.col("ThreePointerAttempts")).alias("ThreePtPct")),
    ((pl.col("OppThreePointersMade")/pl.col("OppThreePointerAttempts")).alias("OppThreePtPct")),
    ((pl.col("FreeThrowsMade")/pl.col("FreeThrowAttempts")).alias("FreeThrowPct")),
    ((pl.col("OppFreeThrowsMade")/pl.col("OppFreeThrowAttempts")).alias("OppFreeThrowPct")),
    pl.when((pl.col("OppConf").is_in(["sec","acc","big_ten","big_twelve","pac_ten","pac_twelve","big_east"])) & (pl.col("WinFlag") == 1)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("Power5WinFlag"),
    pl.when((~pl.col("OppConf").is_in(["sec","acc","big_ten","big_twelve","pac_ten","pac_twelve","big_east"])) & (pl.col("WinFlag") == 1)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("NonPower5WinFlag"),
    pl.when((pl.col("OppConf").is_in(["sec","acc","big_ten","big_twelve","pac_ten","pac_twelve","big_east"])) & (pl.col("LoseFlag") == 1)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("Power5LoseFlag"),
    pl.when((~pl.col("OppConf").is_in(["sec","acc","big_ten","big_twelve","pac_ten","pac_twelve","big_east"])) & (pl.col("LoseFlag") == 1)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("NonPower5LoseFlag")
)

# Join coaching data, making sure only the active coach is listed for each game.
allData = allData.join(mCoaches, on = ["Season","TeamID"], how="left").with_columns(
    pl.col("CoachName").count().over(["Season","TeamID","GameID"]).alias("CoachesDuringSeason")
).filter((pl.col("DayNum") <= pl.col("LastDayNum")) & (pl.col("FirstDayNum") <= pl.col("DayNum")))\
.drop(["FirstDayNum","LastDayNum","CoachesDuringSeason"]).sort(["Season","TeamID","DayNum"])

# Add 2 columns: current coach NCAA tourney record, and current school NCAA tourney record. 
# Need to perform separate calculation if we want to use historical data prior to 2003.
coachWins = pl.concat([
    postSeasonCompact.select(["Season","DayNum","WTeamID"]).rename({"WTeamID":"TeamID"}).with_columns(
        pl.lit(1).alias("TourneyWinFlag"),
        pl.lit(0).alias("TourneyLoseFlag")
    ),
    postSeasonCompact.select(["Season","DayNum","LTeamID"]).rename({"LTeamID":"TeamID"}).with_columns(
        pl.lit(0).alias("TourneyWinFlag"),
        pl.lit(1).alias("TourneyLoseFlag")
    )
    ])\
.join(mCoaches, on = ["Season","TeamID"], how="left")\
.filter((pl.col("DayNum") <= pl.col("LastDayNum")) & (pl.col("FirstDayNum") <= pl.col("DayNum")))\
.sort(["CoachName","Season","DayNum"])\
.with_columns(
    (pl.col("TourneyWinFlag").cum_sum().over(["CoachName"]) - pl.col("TourneyWinFlag")).alias("ActiveTourneyWins_Coach")
).select(["Season","CoachName","DayNum","ActiveTourneyWins_Coach"])


teamWins = pl.concat([
    postSeasonCompact.select(["Season","DayNum","WTeamID"]).rename({"WTeamID":"TeamID"}).with_columns(
        pl.lit(1).alias("TourneyWinFlag"),
        pl.lit(0).alias("TourneyLoseFlag")
    ),
    postSeasonCompact.select(["Season","DayNum","LTeamID"]).rename({"LTeamID":"TeamID"}).with_columns(
        pl.lit(0).alias("TourneyWinFlag"),
        pl.lit(1).alias("TourneyLoseFlag")
    )
    ])\
.join(mCoaches, on = ["Season","TeamID"], how="left")\
.filter((pl.col("DayNum") <= pl.col("LastDayNum")) & (pl.col("FirstDayNum") <= pl.col("DayNum")))\
.sort(["TeamID","Season","DayNum"])\
.with_columns(
    (pl.col("TourneyWinFlag").cum_sum().over(["TeamID"]) - pl.col("TourneyWinFlag")).alias("ActiveTourneyWins_School")
).select(["Season","TeamID","DayNum","ActiveTourneyWins_School"])

# This will fill tourney wins starting with 2003 postseason
allData = allData.join(teamWins, on = ["Season","TeamID","DayNum"], how='left')\
.join(coachWins, on = ["Season","CoachName","DayNum"], how='left')\
.sort(["TeamID","Season","DayNum"])\
.with_columns(
    pl.col("ActiveTourneyWins_School").fill_null(strategy='forward').over("TeamID"),
    pl.col("ActiveTourneyWins_Coach").fill_null(strategy='forward').over("CoachName")
)


# This will fill pre-2003 active team/coach tourney wins
teamWins1989_2002 = teamWins.filter(pl.col("Season").is_between(1989,2003,closed='left')).group_by("TeamID").agg(pl.col("ActiveTourneyWins_School").max().alias("TeamPre2003TourneyWins")).with_columns(pl.lit(2003).alias("Season"))
coachWins1989_2002 = coachWins.filter(pl.col("Season").is_between(1989,2003,closed='left')).group_by("CoachName").agg(pl.col("ActiveTourneyWins_Coach").max().alias("CoachPre2003TourneyWins")).with_columns(pl.lit(2003).alias("Season"))

allData = allData.join(teamWins1989_2002, on = ["Season","TeamID"], how="left").with_columns(
    pl.col("ActiveTourneyWins_School").fill_null(strategy='backward').over("TeamID")
)
allData = allData.join(coachWins1989_2002, on = ["Season","CoachName"], how="left").with_columns(
    pl.col("ActiveTourneyWins_Coach").fill_null(strategy='backward').over("CoachName")
)


# Join seed lists
allData = allData.join(tourneySeeds, on=["Season","TeamID"], how='full').filter(pl.col("GameID").is_not_null()).rename({"Seed":"NCAATourneySeed"})\
    .with_columns(
        pl.when(pl.col("NCAATourneyFlag")==1)
        .then(pl.col("NCAATourneySeed").str.slice(1,2))
        .otherwise(None)
        .cast(pl.Int32)
        .alias("NCAATourneySeed")
    ).drop(["Season_right","TeamID_right"])


# Join AP/NET ranking data for each game
# First define helper function to provide range for eventual explode() function
def ranking_day_range(dateRange):
    return list(range(dateRange[0], dateRange[1]+1))

apRankPeriods = apRankings.select(["Season","RankingDayNum"]).unique().sort(["Season","RankingDayNum"]).rename({"RankingDayNum":"RankPeriodStart"}).with_columns(
    (pl.col("RankPeriodStart").shift(-1) - 1).fill_null(pl.lit(200)).alias("RankPeriodEnd")
).join(apRankings, left_on=["Season","RankPeriodStart"], right_on=["Season","RankingDayNum"], how='left').with_columns(
    pl.concat_list("RankPeriodStart","RankPeriodEnd").map_elements(
        ranking_day_range,
        return_dtype=pl.List(pl.Int64)
    ).alias("DayNum")
).explode("DayNum")


pomRankPeriods = pomRankings.select(["Season","RankingDayNum"]).unique().sort(["Season","RankingDayNum"]).rename({"RankingDayNum":"RankPeriodStart"}).with_columns(
    (pl.col("RankPeriodStart").shift(-1) - 1).fill_null(pl.lit(200)).alias("RankPeriodEnd")
).join(pomRankings, left_on=["Season","RankPeriodStart"], right_on=["Season","RankingDayNum"], how='left').with_columns(
    pl.concat_list("RankPeriodStart","RankPeriodEnd").map_elements(
        ranking_day_range,
        return_dtype=pl.List(pl.Int64)
    ).alias("DayNum")
).explode("DayNum")


netRankPeriods = netRankings.select(["Season","RankingDayNum"]).unique().sort(["Season","RankingDayNum"]).rename({"RankingDayNum":"RankPeriodStart"}).with_columns(
    (pl.col("RankPeriodStart").shift(-1) - 1).fill_null(pl.lit(200)).alias("RankPeriodEnd")
).join(netRankings, left_on=["Season","RankPeriodStart"], right_on=["Season","RankingDayNum"], how='left').with_columns(
    pl.concat_list("RankPeriodStart","RankPeriodEnd").map_elements(
        ranking_day_range,
        return_dtype=pl.List(pl.Int64)
    ).alias("DayNum")
).explode("DayNum")


allData = allData.join(apRankPeriods, on=["Season","TeamID","DayNum"], how='left')\
.drop(["RankPeriodStart","RankPeriodEnd","SystemName"])\
.rename({"OrdinalRank":"ActiveAPRank"})

allData = allData.join(pomRankPeriods, on=["Season","TeamID","DayNum"], how='left')\
.drop(["RankPeriodStart","RankPeriodEnd","SystemName"])\
.rename({"OrdinalRank":"ActivePOMRank"})

allData = allData.join(netRankPeriods, on=["Season","TeamID","DayNum"], how='left')\
.drop(["RankPeriodStart","RankPeriodEnd","SystemName"])\
.rename({"OrdinalRank":"ActiveNETRank"})

# Export data for app predictions
allData.filter(pl.col("Season")==2025).write_parquet('2025_data.parquet')

# Setup rolling average function for game level rolling average statistics
cum_mean = lambda x: pl.col(x).cum_sum().truediv(pl.col(x).cum_count())

# Select & calculate model variables. Impute values using literal values/backward fill/forward fill
modelData = allData.sort(["Season","TeamID","DayNum"]).with_columns(
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
    ((pl.col("WinFlag").cum_sum().shift(1).over(['Season','TeamID']))/((pl.col("WinFlag").cum_sum().shift(1).over(['Season','TeamID']))+(pl.col("LoseFlag").cum_sum().shift(1).over(['Season','TeamID'])))).alias("RollingWinPct_Overall"),
    ((pl.col("WinFlag").rolling_sum(window_size=5).shift(1).over(['Season','TeamID']))/((pl.col("WinFlag").rolling_sum(window_size=5).shift(1).over(['Season','TeamID']))+(pl.col("LoseFlag").rolling_sum(window_size=5).shift(1).over(['Season','TeamID'])))).alias("RollingWinPct_Last5"),
    # Power6 win% 
    ((pl.col("Power5WinFlag").cum_sum().shift(1).over(['Season','TeamID']))/((pl.col("Power5WinFlag").cum_sum().shift(1).over(['Season','TeamID']))+(pl.col("Power5LoseFlag").cum_sum().shift(1).over(['Season','TeamID'])))).alias("RollingP5WinPct_Overall"),
    # Non-power6 win%
    ((pl.col("NonPower5WinFlag").cum_sum().shift(1).over(['Season','TeamID']))/((pl.col("NonPower5WinFlag").cum_sum().shift(1).over(['Season','TeamID']))+(pl.col("NonPower5LoseFlag").cum_sum().shift(1).over(['Season','TeamID'])))).alias("RollingNP5WinPct_Overall"),
    # Points scored
    (cum_mean("PointsScored").shift(1).over(['Season','TeamID']).alias('PointsScoredAvg_Overall')),
    (pl.col("PointsScored").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('PointsScoredAvg_Last5')),
    # Points allowed
    (cum_mean("PointsAllowed").shift(1).over(['Season','TeamID']).alias('PointsAllowedAvg_Overall')),
    (pl.col("PointsAllowed").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('PointsAllowedAvg_Last5')),
    # Rebounds
    (cum_mean("TotalRebounds").shift(1).over(['Season','TeamID']).alias('ReboundsAvg_Overall')),
    (pl.col("TotalRebounds").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('ReboundsAvg_Last5')),
    # Rebounds allowed
    (cum_mean("OppTotalRebounds").shift(1).over(['Season','TeamID']).alias('OppReboundsAvg_Overall')),
    (pl.col("OppTotalRebounds").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OppReboundsAvg_Last5')),
    # Offensive rebounds
    (cum_mean("OffensiveRebounds").shift(1).over(['Season','TeamID']).alias('OffensiveReboundsAvg_Overall')),
    (pl.col("OffensiveRebounds").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OffensiveReboundsAvg_Last5')),
    # Defensive rebounds
    (cum_mean("DefensiveRebounds").shift(1).over(['Season','TeamID']).alias('DefensiveReboundsAvg_Overall')),
    (pl.col("DefensiveRebounds").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('DefensiveReboundsAvg_Last5')),
    # Offensive rebounds allowed
    (cum_mean("OppOffensiveRebounds").shift(1).over(['Season','TeamID']).alias('OppOffensiveReboundsAvg_Overall')),
    (pl.col("OppOffensiveRebounds").shift(1).rolling_mean(window_size=5).over(['Season','TeamID']).alias('OppOffensiveReboundsAvg_Last5')),
    # Defensive rebounds allowed
    (cum_mean("OppDefensiveRebounds").shift(1).over(['Season','TeamID']).alias('OppDefensiveReboundsAvg_Overall')),
    (pl.col("OppDefensiveRebounds").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OppDefensiveReboundsAvg_Last5')),
    # Turnovers given
    (cum_mean("Turnovers").shift(1).over(['Season','TeamID']).alias('TurnoversAvg_Overall')),
    (pl.col("Turnovers").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('TurnoversAvg_Last5')),
    # Turnovers against
    (cum_mean("OppTurnovers").shift(1).over(['Season','TeamID']).alias('OppTurnoversAvg_Overall')),
    (pl.col("OppTurnovers").shift(1).rolling_mean(window_size=5).over(['Season','TeamID']).alias('OppTurnoversAvg_Last5')),
    # Steals
    (cum_mean("Steals").shift(1).over(['Season','TeamID']).alias('StealsAvg_Overall')),
    (pl.col("Steals").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('StealsAvg_Last5')),
    # Steals against
    (cum_mean("OppSteals").shift(1).over(['Season','TeamID']).alias('OppStealsAvg_Overall')),
    (pl.col("OppSteals").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OppStealsAvg_Last5')),
    # Blocks
    (cum_mean("Blocks").shift(1).over(['Season','TeamID']).alias('BlocksAvg_Overall')),
    (pl.col("Blocks").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('BlocksAvg_Last5')),
    # Blocks against
    (cum_mean("OppBlocks").shift(1).over(['Season','TeamID']).alias('OppBlocksAvg_Overall')),
    (pl.col("OppBlocks").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OppBlocksAvg_Last5')),
    # Player fouls
    (cum_mean("PlayerFouls").shift(1).over(['Season','TeamID']).alias('FoulsAvg_Overall')),
    (pl.col("PlayerFouls").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('FoulsAvg_Last5')),
    # Player fouls opponent
    (cum_mean("OppPlayerFouls").shift(1).over(['Season','TeamID']).alias('OppFoulsAvg_Overall')),
    (pl.col("OppPlayerFouls").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OppFoulsAvg_Last5')),
    # Free throw % 
    (cum_mean("FreeThrowPct").shift(1).over(['Season','TeamID']).alias('FreeThrowPct_Overall')),
    (pl.col("FreeThrowPct").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('FreeThrowPct_Last5')),
    # Free throw % opponent
    (cum_mean("OppFreeThrowPct").shift(1).over(['Season','TeamID']).alias('OppFreeThrowPct_Overall')),
    (pl.col("OppFreeThrowPct").rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OppFreeThrowPct_Last5')),
    # Field goal %
    (cum_mean("FieldGoalPct").shift(1).over(['Season','TeamID']).alias('FieldGoalPct_Overall')),
    (pl.col('FieldGoalPct').rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('FieldGoalPct_Last5')),
    # Field goal % allowed
    (cum_mean("OppFieldGoalPct").shift(1).over(['Season','TeamID']).alias('OppFieldGoalPct_Overall')),
    (pl.col('OppFieldGoalPct').rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OppFieldGoalPct_Last5')),
    # 3P%
    (cum_mean("ThreePtPct").shift(1).over(['Season','TeamID']).alias('ThreePtPct_Overall')),
    (pl.col('ThreePtPct').rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('ThreePtPct_Last5')),
    # 3P% allowed
    (cum_mean("OppThreePtPct").shift(1).over(['Season','TeamID']).alias('OppThreePtPct_Overall')),
    (pl.col('OppThreePtPct').rolling_mean(window_size=5).shift(1).over(['Season','TeamID']).alias('OppThreePtPct_Last5'))
).select(["WinFlag","Season","GameID","DayNum","TeamID","RollingWinPct_Overall","RollingWinPct_Last5","GameLocation","TeamConf","RegSeasonFlag","ConfTourneyFlag","NCAATourneyFlag",
          "RollingP5WinPct_Overall","RollingNP5WinPct_Overall",
          "CoachName","ActiveTourneyWins_School","ActiveTourneyWins_Coach","NCAATourneySeed","ActiveAPRank","ActivePOMRank","ActiveNETRank","SeasonBestAPRank","SeasonBestPOMRank",
          "PointsScoredAvg_Overall","PointsScoredAvg_Last5","PointsAllowedAvg_Overall","PointsAllowedAvg_Last5",
          "ReboundsAvg_Overall","ReboundsAvg_Last5","OppReboundsAvg_Overall","OppReboundsAvg_Last5",
          "FreeThrowPct_Overall","FreeThrowPct_Last5","OppFreeThrowPct_Overall","OppFreeThrowPct_Last5",
          "FieldGoalPct_Overall","FieldGoalPct_Last5","OppFieldGoalPct_Overall","OppFieldGoalPct_Last5",
          "ThreePtPct_Overall","ThreePtPct_Last5","OppThreePtPct_Overall","OppThreePtPct_Last5"])

# Median: points, rebounds, offensive rebounds, defensive rebounds,turnovers, steals, blocks, fouls, free throw pct, field goal pct, three point pct
points_filler = allData['PointsScored'].median()
rebounds_filler = allData.with_columns((pl.col("OffensiveRebounds")+pl.col("DefensiveRebounds")).alias("Rebounds")).select("Rebounds").median()
offensive_rebounds_filler = allData['OffensiveRebounds'].median()
defensive_rebounds_filler = allData['DefensiveRebounds'].median()
fgp_filler = allData['FieldGoalPct'].median()
ftp_filler = allData['FreeThrowPct'].median()
tpp_filler = allData['ThreePtPct'].median()
turnovers_filler = allData['Turnovers'].median()
assists_filler = allData['Assists'].median()
steals_filler = allData['Steals'].median()
fouls_filler = allData['PlayerFouls'].median()
blocks_filler = allData['Blocks'].median()

# Filter to remove first five games (mostly null values across columns). Then fill remaining null values
modelData = modelData.sort(["Season","TeamID","DayNum"]).with_columns(
    pl.col("DayNum").rank(method='dense').over(["Season","TeamID"]).alias("GameNumber") 
).filter(pl.col("GameNumber") > 5).fill_null(0).fill_nan(0)



teamA_data = modelData.sort(["GameID","GameLocation"]).with_columns(
    pl.concat_str(pl.col("GameLocation"),pl.col("TeamID")).rank(method='dense').over(["GameID"]).alias("SplitReference")
).filter(pl.col("SplitReference") == 1)
teamA_data.columns = [col + "_A" if col not in ["GameID","Season","DayNum"] else col for col in teamA_data.columns]

teamB_data = modelData.sort(["GameID","GameLocation"]).with_columns(
    pl.concat_str(pl.col("GameLocation"),pl.col("TeamID")).rank(method='dense').over(["GameID"]).alias("SplitReference")
).filter(pl.col("SplitReference") == 2)
teamB_data.columns = [col + "_B" if col not in ["GameID","Season","DayNum"] else col for col in teamB_data.columns]

finalData = teamA_data.join(teamB_data, how='inner', on = ['GameID'])\
    .drop(["WinFlag_B","RegSeasonFlag_B","ConfTourneyFlag_B","NCAATourneyFlag_B","GameNumber_A","GameNumber_B","SplitReference_A","SplitReference_B","GameID","DayNum","Season_right","DayNum_right"])

# Write to parquet file
finalData.write_parquet('model_data.parquet')
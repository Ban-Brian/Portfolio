############################################################
# NFL Weekly Prediction Bot (2025 Season)
# Requires: nflreadr, nflfastR, dplyr, tidyr, xgboost, knitr
############################################################

# --- Libraries ---
library(nflreadr)
library(nflfastR)
library(dplyr)
library(tidyr)
library(xgboost)
library(knitr)

# --- Parameters ---
train_seasons <- 2016:2024   # train on completed seasons
predict_season <- 2025       # current season
set.seed(42)

############################################################
# Feature Engineering Functions
############################################################

# Offense/Defense summary (rolling averages)
team_features <- function(season) {
  pbp <- load_pbp(season)
  
  team_off <- pbp %>%
    group_by(posteam, week) %>%
    summarise(
      off_epa = mean(epa, na.rm = TRUE),
      off_pass_rate = mean(pass, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(posteam) %>%
    arrange(week) %>%
    mutate(
      off_epa_roll = zoo::rollapply(off_epa, 3, mean, fill = NA, align = "right"),
      off_pass_rate_roll = zoo::rollapply(off_pass_rate, 3, mean, fill = NA, align = "right")
    ) %>%
    ungroup() %>%
    rename(team = posteam)
  
  team_def <- pbp %>%
    group_by(defteam, week) %>%
    summarise(
      def_epa = mean(epa, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(defteam) %>%
    arrange(week) %>%
    mutate(
      def_epa_roll = zoo::rollapply(def_epa, 3, mean, fill = NA, align = "right")
    ) %>%
    ungroup() %>%
    rename(team = defteam)
  
  full_join(team_off, team_def, by = c("team", "week"))
}

# Injuries summary
injury_summary <- function(season) {
  inj <- load_injuries(season)
  inj %>%
    filter(report_status == "Out") %>%
    group_by(season, week, team) %>%
    summarise(starters_out = n(), .groups = "drop")
}

############################################################
# Training Data
############################################################

train_games <- load_schedules(train_seasons) %>%
  filter(season_type == "REG") %>%
  select(season, week, game_id, home_team, away_team,
         home_score, away_score) %>%
  mutate(result = ifelse(home_score > away_score, 1, 0))

# Join features
feat_list <- lapply(train_seasons, team_features)
inj_list  <- lapply(train_seasons, injury_summary)

train_feats <- bind_rows(feat_list)
train_inj   <- bind_rows(inj_list)

train_games <- train_games %>%
  left_join(train_feats, by = c("home_team" = "team", "week", "season")) %>%
  rename_with(~ paste0("home_", .), starts_with("off_")) %>%
  rename_with(~ paste0("home_", .), starts_with("def_")) %>%
  left_join(train_feats, by = c("away_team" = "team", "week", "season")) %>%
  rename_with(~ paste0("away_", .), starts_with("off_")) %>%
  rename_with(~ paste0("away_", .), starts_with("def_")) %>%
  left_join(train_inj, by = c("home_team" = "team", "week", "season")) %>%
  rename(home_injuries = starters_out) %>%
  left_join(train_inj, by = c("away_team" = "team", "week", "season")) %>%
  rename(away_injuries = starters_out)

train_matrix <- train_games %>%
  select(home_off_epa_roll, home_off_pass_rate_roll, home_def_epa_roll,
         away_off_epa_roll, away_off_pass_rate_roll, away_def_epa_roll,
         home_injuries, away_injuries) %>%
  as.matrix()

dtrain <- xgb.DMatrix(data = train_matrix, label = train_games$result)

# Train model
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 4,
  eta = 0.1
)
model <- xgboost(params = params, data = dtrain, nrounds = 100, verbose = 0)

############################################################
# Auto-Detect Next Week
############################################################

sched_curr <- load_schedules(predict_season) %>%
  filter(season_type == "REG")

last_completed_week <- sched_curr %>%
  filter(!is.na(home_score), !is.na(away_score)) %>%
  summarise(last_week = max(week, na.rm = TRUE)) %>%
  pull(last_week)

predict_week <- last_completed_week + 1
cat("📅 Predicting Week", predict_week, "of", predict_season, "\n")

############################################################
# Prediction Data
############################################################

sched_pred <- sched_curr %>%
  filter(week == predict_week) %>%
  select(season, week, game_id, home_team, away_team)

pred_feats <- team_features(predict_season)
pred_inj   <- injury_summary(predict_season)

pred_games <- sched_pred %>%
  left_join(pred_feats, by = c("home_team" = "team", "week", "season")) %>%
  rename_with(~ paste0("home_", .), starts_with("off_")) %>%
  rename_with(~ paste0("home_", .), starts_with("def_")) %>%
  left_join(pred_feats, by = c("away_team" = "team", "week", "season")) %>%
  rename_with(~ paste0("away_", .), starts_with("off_")) %>%
  rename_with(~ paste0("away_", .), starts_with("def_")) %>%
  left_join(pred_inj, by = c("home_team" = "team", "week", "season")) %>%
  rename(home_injuries = starters_out) %>%
  left_join(pred_inj, by = c("away_team" = "team", "week", "season")) %>%
  rename(away_injuries = starters_out)

pred_matrix <- pred_games %>%
  select(home_off_epa_roll, home_off_pass_rate_roll, home_def_epa_roll,
         away_off_epa_roll, away_off_pass_rate_roll, away_def_epa_roll,
         home_injuries, away_injuries) %>%
  as.matrix()

pred_probs <- predict(model, xgb.DMatrix(pred_matrix))

pred_results <- pred_games %>%
  mutate(
    prob_home = pred_probs,
    prob_away = 1 - pred_probs,
    pred_winner = ifelse(prob_home > 0.5, home_team, away_team)
  ) %>%
  select(season, week, home_team, away_team, prob_home, prob_away, pred_winner)

############################################################
# Output Predictions
############################################################

print(pred_results %>% knitr::kable(digits = 2))


############################################################
# Enhanced NFL Weekly Prediction Bot (2025 Season)
# Requires: nflreadr, nflfastR, dplyr, tidyr, xgboost, ggplot2, zoo
############################################################

library(nflreadr)
library(nflfastR)
library(dplyr)
library(tidyr)
library(xgboost)
library(ggplot2)
library(zoo)

set.seed(42)

############################################################
# Enhanced Feature Engineering
############################################################

team_features <- function(seas) {
  pbp <- load_pbp(seas)
  
  # Offensive features
  team_off <- pbp %>%
    filter(!is.na(epa), !is.na(posteam)) %>%
    group_by(posteam, week) %>%
    summarise(
      off_epa = mean(epa, na.rm = TRUE),
      off_pass_epa = mean(epa[pass == 1], na.rm = TRUE),
      off_rush_epa = mean(epa[rush == 1], na.rm = TRUE),
      off_success_rate = mean(success, na.rm = TRUE),
      off_explosive_rate = mean(epa > 0.5, na.rm = TRUE),
      off_third_down_conv = mean(third_down_converted, na.rm = TRUE),
      off_red_zone_epa = mean(epa[yardline_100 <= 20], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(posteam) %>%
    arrange(week) %>%
    mutate(
      off_epa_roll = rollapply(off_epa, width = 4, FUN = mean, fill = NA, align = "right"),
      off_pass_epa_roll = rollapply(off_pass_epa, width = 4, FUN = mean, fill = NA, align = "right"),
      off_rush_epa_roll = rollapply(off_rush_epa, width = 4, FUN = mean, fill = NA, align = "right"),
      off_success_roll = rollapply(off_success_rate, width = 4, FUN = mean, fill = NA, align = "right"),
      off_explosive_roll = rollapply(off_explosive_rate, width = 4, FUN = mean, fill = NA, align = "right"),
      off_third_down_roll = rollapply(off_third_down_conv, width = 4, FUN = mean, fill = NA, align = "right"),
      off_rz_epa_roll = rollapply(off_red_zone_epa, width = 4, FUN = mean, fill = NA, align = "right"),
      season = seas
    ) %>%
    ungroup() %>%
    rename(team = posteam)
  
  # Defensive features
  team_def <- pbp %>%
    filter(!is.na(epa), !is.na(defteam)) %>%
    group_by(defteam, week) %>%
    summarise(
      def_epa = mean(epa, na.rm = TRUE),
      def_pass_epa = mean(epa[pass == 1], na.rm = TRUE),
      def_rush_epa = mean(epa[rush == 1], na.rm = TRUE),
      def_success_rate = mean(success, na.rm = TRUE),
      def_pressure_rate = mean(qb_hit | sack, na.rm = TRUE),
      def_third_down_stop = 1 - mean(third_down_converted, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(defteam) %>%
    arrange(week) %>%
    mutate(
      def_epa_roll = rollapply(def_epa, width = 4, FUN = mean, fill = NA, align = "right"),
      def_pass_epa_roll = rollapply(def_pass_epa, width = 4, FUN = mean, fill = NA, align = "right"),
      def_rush_epa_roll = rollapply(def_rush_epa, width = 4, FUN = mean, fill = NA, align = "right"),
      def_success_roll = rollapply(def_success_rate, width = 4, FUN = mean, fill = NA, align = "right"),
      def_pressure_roll = rollapply(def_pressure_rate, width = 4, FUN = mean, fill = NA, align = "right"),
      def_third_stop_roll = rollapply(def_third_down_stop, width = 4, FUN = mean, fill = NA, align = "right"),
      season = seas
    ) %>%
    ungroup() %>%
    rename(team = defteam)
  
  full_join(team_off, team_def, by = c("team", "week", "season"))
}

############################################################
# Build Training Data
############################################################

cat("Building training data...\n")
train_seasons <- 2016:2024
predict_season <- 2025

train_games <- load_schedules(train_seasons) %>%
  filter(game_type == "REG", !is.na(home_score), !is.na(away_score)) %>%
  select(season, week, game_id, home_team, away_team, home_score, away_score, 
         spread_line, total_line) %>%
  mutate(result = ifelse(home_score > away_score, 1, 0))

# Build features for all training seasons
train_feats_list <- list()
train_inj_list <- list()

for (s in train_seasons) {
  cat("Processing season", s, "...\n")
  train_feats_list[[as.character(s)]] <- team_features(s)
  
  # Try to get injury data, skip if not available
  inj_data <- tryCatch({
    injury_summary(s)
  }, error = function(e) {
    cat("  Warning: Injury data not available for", s, "\n")
    data.frame(week = integer(), team = character(), 
               injury_impact = numeric(), season = integer())
  })
  train_inj_list[[as.character(s)]] <- inj_data
}

train_feats <- bind_rows(train_feats_list)
train_inj <- bind_rows(train_inj_list)

# Only join injury data if we have any
if (nrow(train_inj) > 0) {
  train_games <- train_games %>%
    left_join(train_feats, by = c("season", "week", "home_team" = "team")) %>%
    rename_with(~ paste0("home_", .), c(off_epa_roll, off_pass_epa_roll, off_rush_epa_roll, 
                                        off_success_roll, off_explosive_roll, off_third_down_roll,
                                        off_rz_epa_roll, def_epa_roll, def_pass_epa_roll,
                                        def_rush_epa_roll, def_success_roll, def_pressure_roll,
                                        def_third_stop_roll)) %>%
    left_join(train_feats, by = c("season", "week", "away_team" = "team")) %>%
    rename_with(~ paste0("away_", .), c(off_epa_roll, off_pass_epa_roll, off_rush_epa_roll,
                                        off_success_roll, off_explosive_roll, off_third_down_roll,
                                        off_rz_epa_roll, def_epa_roll, def_pass_epa_roll,
                                        def_rush_epa_roll, def_success_roll, def_pressure_roll,
                                        def_third_stop_roll)) %>%
    left_join(train_inj, by = c("season", "week", "home_team" = "team")) %>%
    rename(home_injuries = injury_impact) %>%
    left_join(train_inj, by = c("season", "week", "away_team" = "team")) %>%
    rename(away_injuries = injury_impact) %>%
    replace_na(list(home_injuries = 0, away_injuries = 0))
} else {
  train_games <- train_games %>%
    left_join(train_feats, by = c("season", "week", "home_team" = "team")) %>%
    rename_with(~ paste0("home_", .), c(off_epa_roll, off_pass_epa_roll, off_rush_epa_roll, 
                                        off_success_roll, off_explosive_roll, off_third_down_roll,
                                        off_rz_epa_roll, def_epa_roll, def_pass_epa_roll,
                                        def_rush_epa_roll, def_success_roll, def_pressure_roll,
                                        def_third_stop_roll)) %>%
    left_join(train_feats, by = c("season", "week", "away_team" = "team")) %>%
    rename_with(~ paste0("away_", .), c(off_epa_roll, off_pass_epa_roll, off_rush_epa_roll,
                                        off_success_roll, off_explosive_roll, off_third_down_roll,
                                        off_rz_epa_roll, def_epa_roll, def_pass_epa_roll,
                                        def_rush_epa_roll, def_success_roll, def_pressure_roll,
                                        def_third_stop_roll)) %>%
    mutate(home_injuries = 0, away_injuries = 0)
}

feature_cols <- c(
  "home_off_epa_roll", "home_off_pass_epa_roll", "home_off_rush_epa_roll",
  "home_off_success_roll", "home_off_explosive_roll", "home_off_third_down_roll",
  "home_off_rz_epa_roll", "home_def_epa_roll", "home_def_pass_epa_roll",
  "home_def_rush_epa_roll", "home_def_success_roll", "home_def_pressure_roll",
  "home_def_third_stop_roll", "away_off_epa_roll", "away_off_pass_epa_roll",
  "away_off_rush_epa_roll", "away_off_success_roll", "away_off_explosive_roll",
  "away_off_third_down_roll", "away_off_rz_epa_roll", "away_def_epa_roll",
  "away_def_pass_epa_roll", "away_def_rush_epa_roll", "away_def_success_roll",
  "away_def_pressure_roll", "away_def_third_stop_roll", "home_injuries", "away_injuries"
)

train_clean <- train_games %>%
  filter(complete.cases(select(., all_of(feature_cols))))

train_matrix <- train_clean %>%
  select(all_of(feature_cols)) %>%
  as.matrix()

dtrain <- xgb.DMatrix(data = train_matrix, label = train_clean$result)

############################################################
# Train Model with Cross-Validation
############################################################

cat("Training model...\n")
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3
)

cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 300,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 0
)

best_nrounds <- cv_results$best_iteration
model <- xgboost(params = params, data = dtrain, nrounds = best_nrounds, verbose = 0)

cat("Model trained with", best_nrounds, "rounds\n")
cat("CV Log Loss:", min(cv_results$evaluation_log$test_logloss_mean), "\n\n")

############################################################
# Auto-Detect Next Week
############################################################

sched_curr <- load_schedules(predict_season) %>%
  filter(game_type == "REG")

last_completed_week <- sched_curr %>%
  filter(!is.na(home_score), !is.na(away_score)) %>%
  summarise(last_week = max(week, na.rm = TRUE)) %>%
  pull(last_week)

predict_week <- ifelse(length(last_completed_week) == 0 || is.na(last_completed_week), 
                       1, last_completed_week + 1)

cat("Predicting Week", predict_week, "of", predict_season, "\n\n")

############################################################
# Prediction
############################################################

sched_pred <- sched_curr %>%
  filter(week == predict_week) %>%
  select(season, week, game_id, home_team, away_team, gameday, gametime, spread_line)

if (nrow(sched_pred) == 0) {
  cat("No games scheduled for Week", predict_week, "\n")
} else {
  pred_feats <- team_features(predict_season)
  
  # Try to get injury data, use empty dataframe if not available
  pred_inj <- tryCatch({
    injury_summary(predict_season)
  }, error = function(e) {
    cat("Note: Injury data not available for", predict_season, "- predictions without injury info\n")
    data.frame(week = integer(), team = character(), 
               injury_impact = numeric(), season = integer())
  })
  
  pred_games <- sched_pred %>%
    left_join(pred_feats, by = c("season", "week", "home_team" = "team")) %>%
    rename_with(~ paste0("home_", .), c(off_epa_roll, off_pass_epa_roll, off_rush_epa_roll,
                                        off_success_roll, off_explosive_roll, off_third_down_roll,
                                        off_rz_epa_roll, def_epa_roll, def_pass_epa_roll,
                                        def_rush_epa_roll, def_success_roll, def_pressure_roll,
                                        def_third_stop_roll)) %>%
    left_join(pred_feats, by = c("season", "week", "away_team" = "team")) %>%
    rename_with(~ paste0("away_", .), c(off_epa_roll, off_pass_epa_roll, off_rush_epa_roll,
                                        off_success_roll, off_explosive_roll, off_third_down_roll,
                                        off_rz_epa_roll, def_epa_roll, def_pass_epa_roll,
                                        def_rush_epa_roll, def_success_roll, def_pressure_roll,
                                        def_third_stop_roll))
  
  # Only join injury data if available
  if (nrow(pred_inj) > 0) {
    pred_games <- pred_games %>%
      left_join(pred_inj, by = c("season", "week", "home_team" = "team")) %>%
      rename(home_injuries = injury_impact) %>%
      left_join(pred_inj, by = c("season", "week", "away_team" = "team")) %>%
      rename(away_injuries = injury_impact)
  } else {
    pred_games <- pred_games %>%
      mutate(home_injuries = 0, away_injuries = 0)
  }
  
  pred_games <- pred_games %>%
    replace_na(list(home_injuries = 0, away_injuries = 0))
  
  pred_matrix <- pred_games %>%
    select(all_of(feature_cols)) %>%
    as.matrix()
  
  pred_probs <- predict(model, xgb.DMatrix(pred_matrix))
  
  pred_results <- pred_games %>%
    mutate(
      prob_home = pred_probs * 100,
      prob_away = (1 - pred_probs) * 100,
      pred_winner = ifelse(pred_probs > 0.5, home_team, away_team),
      confidence = abs(pred_probs - 0.5) * 200,
      matchup = paste(away_team, "@", home_team)
    ) %>%
    select(matchup, gameday, prob_home, prob_away, pred_winner, confidence, spread_line) %>%
    arrange(desc(confidence))
  
  ############################################################
  # Visualizations
  ############################################################
  
  # Win probability chart
  plot_data <- pred_results %>%
    mutate(game_num = row_number()) %>%
    select(matchup, prob_home, prob_away, game_num) %>%
    pivot_longer(cols = c(prob_home, prob_away), names_to = "team_type", values_to = "probability")
  
  p1 <- ggplot(plot_data, aes(x = reorder(matchup, game_num), y = probability, fill = team_type)) +
    geom_col(position = "stack", width = 0.7) +
    geom_hline(yintercept = 50, linetype = "dashed", color = "white", alpha = 0.5) +
    coord_flip() +
    scale_fill_manual(values = c("prob_home" = "#2E7D32", "prob_away" = "#C62828"),
                      labels = c("Home", "Away")) +
    labs(title = paste("Week", predict_week, "Win Probabilities"),
         subtitle = paste(predict_season, "NFL Season"),
         x = NULL, y = "Win Probability (%)", fill = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top",
          plot.title = element_text(face = "bold", size = 16),
          panel.grid.major.y = element_blank())
  
  print(p1)
  
  # Confidence levels
  p2 <- ggplot(pred_results, aes(x = reorder(matchup, confidence), y = confidence)) +
    geom_col(aes(fill = confidence), width = 0.7) +
    geom_text(aes(label = sprintf("%.1f%%", confidence)), hjust = -0.1, size = 3.5) +
    coord_flip() +
    scale_fill_gradient2(low = "#FFA726", mid = "#66BB6A", high = "#1E88E5",
                         midpoint = 50, guide = "none") +
    labs(title = "Prediction Confidence Levels",
         subtitle = "Higher values indicate more confident predictions",
         x = NULL, y = "Confidence (%)") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold", size = 14),
          panel.grid.major.y = element_blank()) +
    expand_limits(y = c(0, 100))
  
  print(p2)
  
  ############################################################
  # Output Table
  ############################################################
  
  cat("\n")
  cat("=" , rep("=", 80), "\n", sep = "")
  cat("  WEEK", predict_week, "PREDICTIONS\n")
  cat("=" , rep("=", 80), "\n", sep = "")
  cat("\n")
  
  for (i in 1:nrow(pred_results)) {
    row <- pred_results[i, ]
    cat(sprintf("%-30s | %s\n", row$matchup, row$gameday))
    cat(sprintf("  Winner: %-10s | Confidence: %.1f%%\n", 
                row$pred_winner, row$confidence))
    cat(sprintf("  Home: %.1f%%  Away: %.1f%%", 
                row$prob_home, row$prob_away))
    if (!is.na(row$spread_line)) {
      cat(sprintf("  | Vegas Line: %.1f", row$spread_line))
    }
    cat("\n\n")
  }
  
  cat("=" , rep("=", 80), "\n", sep = "")
  
  ############################################################
  # Save outputs
  ############################################################
  
  # Save predictions to CSV
  write.csv(pred_results, 
            paste0("nfl_predictions_week_", predict_week, "_", Sys.Date(), ".csv"),
            row.names = FALSE)
  
  # Save plots
  ggsave(paste0("win_probabilities_week_", predict_week, ".png"), 
         plot = p1, width = 10, height = 8, dpi = 300)
  ggsave(paste0("confidence_levels_week_", predict_week, ".png"), 
         plot = p2, width = 10, height = 8, dpi = 300)
  
  cat("\nOutputs saved:\n")
  cat("- Predictions CSV\n")
  cat("- Win probabilities chart (PNG)\n")
  cat("- Confidence levels chart (PNG)\n")
}

cat("\nScript will auto-update predictions each week when new data is available.\n")
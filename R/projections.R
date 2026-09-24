#!/usr/bin/env Rscript
# HANDOFF §3.1 / §6.2 — multi-source projections and rest-of-season ECR, scored in
# Mega Bowl's half-PPR, written where the Python side can read them.
#
#   Rscript R/projections.R [--week N] [--season YYYY] [--out data/build]
#
# Scrapers break mid-season; that is normal and expected. A source that fails is logged
# and skipped, and the run still succeeds as long as SOME source came back. Requiring all
# of them would mean the one week a site changes its markup is the week Chris gets nothing.

suppressWarnings(suppressMessages({
  lib <- path.expand("~/Library/R/arm64/4.5/library")
  if (dir.exists(lib)) .libPaths(c(lib, .libPaths()))
  library(ffanalytics)
}))

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
season  <- as.integer(getarg("--season", format(Sys.Date(), "%Y")))
week    <- suppressWarnings(as.integer(getarg("--week", NA)))
outdir  <- getarg("--out", "data/build")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# Mega Bowl half-PPR (§1). Confirmed with Chris 2026-09-23: two-point conversions score 2,
# and interceptions are -1 rather than the package default of -3.
#
# Built by overriding the package's OWN default object rather than writing a list from
# scratch. projections_table() reads fields a hand-built list does not know to include
# (`all_pos` on each block, the bonus thresholds) and dies on the first one it misses —
# which is exactly what a from-scratch list did here.
scoring <- ffanalytics::scoring
scoring$pass$pass_yds  <- 0.04
scoring$pass$pass_tds  <- 4
scoring$pass$pass_int  <- -1
scoring$rush$rush_yds  <- 0.10
scoring$rush$rush_tds  <- 6
scoring$rec$rec        <- 0.5      # half-PPR
scoring$rec$rec_yds    <- 0.10
scoring$rec$rec_tds    <- 6
scoring$misc$fumbles_lost <- -2
scoring$misc$two_pts      <- 2

SRC <- c("FantasyPros", "CBS", "ESPN", "NFL", "FFToday", "NumberFire")
POS <- c("QB", "RB", "WR", "TE")

cat(sprintf("[proj] season=%s week=%s sources=%s\n", season,
            ifelse(is.na(week), "season", week), paste(SRC, collapse = ",")))

scrape <- NULL
try({
  scrape <- scrape_data(src = SRC, pos = POS, season = season,
                        week = if (is.na(week)) 0 else week)
}, silent = FALSE)

ok <- FALSE
if (!is.null(scrape) && length(scrape)) {
  got <- names(scrape)
  cat("[proj] sources returned:", paste(got, collapse = ", "), "\n")
  tbl <- NULL
  try({
    tbl <- projections_table(scrape, scoring_rules = scoring, avg_type = "robust")
  }, silent = FALSE)
  if (!is.null(tbl) && nrow(tbl)) {
    # §3.1 requires at least two sources per player before a consensus is trustworthy.
    if ("avg_type" %in% names(tbl)) tbl <- tbl[tbl$avg_type == "robust", ]
    path <- file.path(outdir, sprintf("proj_%s_wk%s.csv", season,
                                      ifelse(is.na(week), "ros", sprintf("%02d", week))))
    write.csv(tbl, path, row.names = FALSE)
    cat(sprintf("[proj] %d rows -> %s\n", nrow(tbl), path))
    ok <- TRUE
  } else cat("[proj] projections_table produced nothing\n")
} else cat("[proj] every source failed\n")

# §6.2 rest-of-season ECR, half-PPR. This is what league-mates actually see.
ecr <- NULL
try({
  ecr <- scrape_ecr(rank_period = "ros", position = "Overall", rank_type = "Half")
}, silent = FALSE)
if (!is.null(ecr) && nrow(ecr)) {
  path <- file.path(outdir, sprintf("ecr_ros_%s.csv", season))
  write.csv(ecr, path, row.names = FALSE)
  cat(sprintf("[ecr] %d rows -> %s\n", nrow(ecr), path))
  ok <- TRUE
} else cat("[ecr] unavailable\n")

quit(status = if (ok) 0 else 1)

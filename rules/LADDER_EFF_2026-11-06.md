{
 "lineage": "teacher47",
 "arch": "auto",
 "games": 240,
 "versions": [
  {
   "name": "teacher47_100h_v2",
   "elo": 1426.6,
   "games": 324,
   "win_rate": 7.1,
   "avg_pt": -1328.5
  },
  {
   "name": "teacher47_eff420",
   "elo": 1529.3,
   "games": 318,
   "win_rate": 16.0,
   "avg_pt": 433.0
  },
  {
   "name": "teacher47_eff_dagger",
   "elo": 1544.2,
   "games": 318,
   "win_rate": 17.3,
   "avg_pt": 332.5
  }
 ],
 "gates": [
  {
   "prev": "teacher47_100h_v2",
   "cur": "teacher47_eff420",
   "n": 386,
   "mean_diff": 1826.0,
   "ci_lo": 1058.8,
   "ci_hi": 2593.2,
   "elo_delta": 102.7,
   "elo_band": 44.7,
   "verdict": "PASS"
  },
  {
   "prev": "teacher47_eff420",
   "cur": "teacher47_eff_dagger",
   "n": 378,
   "mean_diff": -269.8,
   "ci_lo": -1213.7,
   "ci_hi": 674.2,
   "elo_delta": 14.9,
   "elo_band": 44.7,
   "verdict": "HOLD"
  }
 ]
}
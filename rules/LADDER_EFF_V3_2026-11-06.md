{
 "lineage": "teacher47",
 "arch": "auto",
 "games": 240,
 "versions": [
  {
   "name": "teacher47_eff420",
   "elo": 1470.8,
   "games": 324,
   "win_rate": 13.9,
   "avg_pt": -777.4
  },
  {
   "name": "teacher47_eff_dagger",
   "elo": 1522.6,
   "games": 318,
   "win_rate": 16.0,
   "avg_pt": 781.2
  },
  {
   "name": "teacher47_eff_v3",
   "elo": 1506.6,
   "games": 318,
   "win_rate": 15.7,
   "avg_pt": -665.3
  }
 ],
 "gates": [
  {
   "prev": "teacher47_eff420",
   "cur": "teacher47_eff_dagger",
   "n": 386,
   "mean_diff": 1705.1,
   "ci_lo": -404.1,
   "ci_hi": 3814.3,
   "elo_delta": 51.8,
   "elo_band": 44.7,
   "verdict": "HOLD"
  },
  {
   "prev": "teacher47_eff_dagger",
   "cur": "teacher47_eff_v3",
   "n": 378,
   "mean_diff": -1649.7,
   "ci_lo": -3832.6,
   "ci_hi": 533.3,
   "elo_delta": -16.0,
   "elo_band": 44.7,
   "verdict": "HOLD"
  }
 ]
}
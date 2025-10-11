export type PlayerRole = 'Batsman' | 'Bowler' | 'All-Rounder' | 'Wicket-Keeper';

export interface Player {
  id: string;
  name: string;
  role: PlayerRole;
  team: string;
  avatar: string;
  predictedPoints: number;
  // consistency: number;
  // form: number;
  // stats: {
  //   matches: number;
  //   runs?: number;
  //   wickets?: number;
  //   average: number;
  //   strikeRate?: number;
  //   economy?: number;
  // };
  // recentForm: number[]; // Last 5 match scores
  // shap: {
  //   runs: number;
  //   // consistency: number;
  //   // form: number;
  // };
}

// export interface TeamComposition {
//   wicketKeeper: Player[];
//   batsmen: Player[];
//   allRounders: Player[];
//   bowlers: Player[];
// }

export interface TeamKPI {
  // consistency: number;
  // diversity: number;
  // form: number;
  totalScore: number;
  deltas: {
    // consistency: number;
    // diversity: number;
    // form: number;
    totalScore: number;
  };
}

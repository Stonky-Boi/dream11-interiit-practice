import { useNavigate } from "react-router-dom";
import { Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import Shuffle from "@/components/ui/Shuffle";

const upcomingMatches = [
  {
    id: "1433370",
    team1: { name: "ENGLAND", abbr: "ENG", color: "bg-green-600" },
    team2: { name: "West Indies", abbr: "WI", color: "bg-red-500" },
    date: "31-10-2024",
    league: "ODI",
  },
  {
    id: "1385701",
    team1: { name: "Australia", abbr: "AUS", color: "bg-yellow-600" },
    team2: { name: "England", abbr: "ENG", color: "bg-green-800" },
    date: "21-09-2024",
    league: "ODI",
  },
  {
    id: "1442992",
    team1: { name: "India", abbr: "IND", color: "bg-blue-600" },
    team2: {
      name: "Sri Lanka",
      abbr: "SL",
      color: "bg-red-700",
    },
    date: "07-08-2024",
    league: "ODI",
  },
  {
    id: "1469168",
    team1: { name: "Australia", abbr: "AUS", color: "bg-yellow-600" },
    team2: { name: "Sri Lanka", abbr: "SL", color: "bg-cyan-600" },
    date: "14-02-2025",
    league: "ODI",
  },
  {
    id: "1476985",
    team1: { name: "South Africa", abbr: "SA", color: "bg-orange-600" },
    team2: { name: "Sri Lanka", abbr: "SL", color: "bg-cyan-600" },
    date: "09-05-2025",
    league: "ODI",
  },
  {
    id: "1454391",
    team1: { name: "INDIA", abbr: "IND", color: "bg-blue-600" },
    team2: { name: "New Zealand", abbr: "NZ", color: "bg-purple-600" },
    date: "24-10-2024",
    league: "ODI",
  },
];

const MatchSelection = () => {
  const navigate = useNavigate();

  const handleMatchSelect = (match: (typeof upcomingMatches)[0]) => {
    navigate("/team-builder", {
      state: {
        match: `${match.team1.name} vs ${match.team2.name}`,
        date: match.date,
        time: match.time,
        team1: match.team1.name,
        team2: match.team2.name,
        league: match.league,
      },
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-8 relative">
      {/* Background Image */}
      <div
        className="fixed inset-0 -z-10"
        style={{
          backgroundImage: "url(public/bgimage.jpg)",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-b from-background/85 via-background/75 to-background/85" />
      </div>
      <div className="w-full max-w-4xl">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            {/* <Sparkles className="w-12 h-12 text-primary animate-glow-pulse" /> */}
            <h1 className="text-5xl font-bold">
              <Shuffle
                text="Dream11 Predictor"
                shuffleDirection="right"
                duration={0.35}
                animationMode="evenodd"
                shuffleTimes={1}
                ease="power3.out"
                stagger={0.03}
                threshold={0.1}
                triggerOnce={true}
                triggerOnHover={true}
                respectReducedMotion={true}
              />
            </h1>
          </div>
          <p className="text-xl text-muted-foreground">
            Select a match to get AI-powered best 11 predictions
          </p>
        </motion.div>

        {/* Match Cards Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
          {upcomingMatches.map((match, index) => (
            <motion.div
              key={match.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => handleMatchSelect(match)}
              className="bg-card/50 backdrop-blur-sm border border-border rounded-xl p-6 cursor-pointer hover:border-primary hover:shadow-lg hover:scale-105 transition-all duration-300"
            >
              {/* Match Number & Date */}
              <div className="flex justify-between items-center mb-6">
                <span className="text-sm text-muted-foreground font-medium">
                  Match {match.id}
                </span>
                <span className="text-sm text-muted-foreground">
                  {match.date}
                </span>
              </div>

              {/* Teams */}
              <div className="flex items-center justify-between mb-6">
                {/* Team 1 */}
                <div className="flex flex-col items-center gap-2 flex-1">
                  <div
                    className={`w-16 h-16 rounded-full ${match.team1.color} flex items-center justify-center border-4 border-background shadow-lg`}
                  >
                    <span className="text-white font-bold text-sm">
                      {match.team1.abbr}
                    </span>
                  </div>
                  <span className="text-xs text-center font-medium">
                    {match.team1.abbr}
                  </span>
                </div>

                {/* VS */}
                <div className="flex-shrink-0 px-4">
                  <span className="text-xl font-bold text-muted-foreground">
                    Vs
                  </span>
                </div>

                {/* Team 2 */}
                <div className="flex flex-col items-center gap-2 flex-1">
                  <div
                    className={`w-16 h-16 rounded-full ${match.team2.color} flex items-center justify-center border-4 border-background shadow-lg`}
                  >
                    <span className="text-white font-bold text-sm">
                      {match.team2.abbr}
                    </span>
                  </div>
                  <span className="text-xs text-center font-medium">
                    {match.team2.abbr}
                  </span>
                </div>
              </div>

              {/* Time */}
              <div className="text-center">
                <span className="text-sm text-muted-foreground">
                  {match.time}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MatchSelection;

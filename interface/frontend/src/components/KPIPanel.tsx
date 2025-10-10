import { motion } from 'framer-motion';
import { TeamKPI, Player } from '@/types/player';
import { TrendingUp, TrendingDown, Award, Users, Flame, Trophy, FileText, Users2 } from 'lucide-react';
import { Separator } from '@/components/ui/separator';

interface KPIPanelProps {
  kpi: TeamKPI;
  lineup: (Player | null)[];
}

export const KPIPanel = ({ kpi, lineup }: KPIPanelProps) => {
  const validPlayers = lineup.filter((p): p is Player => p !== null);
  
  const composition = {
    batsmen: validPlayers.filter(p => p.role === 'Batsman').length,
    bowlers: validPlayers.filter(p => p.role === 'Bowler').length,
    allRounders: validPlayers.filter(p => p.role === 'All-Rounder').length,
    wicketKeepers: validPlayers.filter(p => p.role === 'Wicket-Keeper').length,
  };

  return (
    <div className="w-80 bg-card border-l border-border p-4 overflow-y-auto">
      <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
        <Trophy className="w-5 h-5 text-gold" />
        Team Metrics
      </h2>

      {/* Total Score */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="mb-8 p-6 rounded-xl gold-gradient text-primary-foreground"
      >
        <p className="text-sm font-medium mb-2">Total Team Score</p>
        <div className="flex items-end gap-3">
          <span className="text-5xl font-bold">{kpi.totalScore.toFixed(0)}</span>
          <div className={`flex items-center gap-1 mb-2 ${
            kpi.deltas.totalScore >= 0 ? 'text-secondary' : 'text-destructive'
          }`}>
            {kpi.deltas.totalScore >= 0 ? (
              <TrendingUp className="w-5 h-5" />
            ) : (
              <TrendingDown className="w-5 h-5" />
            )}
            <span className="font-bold">
              {Math.abs(kpi.deltas.totalScore).toFixed(1)}%
            </span>
          </div>
        </div>
      </motion.div>

      <Separator className="my-6" />

      {/* Team Composition */}
      <div className="mb-6">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Users2 className="w-5 h-5 text-gold" />
          Team Composition
        </h3>
        <div className="space-y-3">
          {[
            { label: 'Batsmen', count: composition.batsmen, color: 'bg-blue-500' },
            { label: 'Bowlers', count: composition.bowlers, color: 'bg-green-500' },
            { label: 'All-Rounders', count: composition.allRounders, color: 'bg-purple-500' },
            { label: 'Wicket-Keepers', count: composition.wicketKeepers, color: 'bg-amber-500' },
          ].map((item, index) => (
            <motion.div
              key={item.label}
              initial={{ x: 50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: index * 0.1 + 0.4 }}
              className="flex items-center justify-between p-3 bg-muted/30 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${item.color}`} />
                <span className="text-sm font-medium">{item.label}</span>
              </div>
              <span className="text-lg font-bold">{item.count}</span>
            </motion.div>
          ))}
        </div>
      </div>

      <Separator className="my-6" />

      {/* Team Analysis Report */}
      <div className="mb-6">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <FileText className="w-5 h-5 text-gold" />
          Team Analysis Report
        </h3>
        <div className="space-y-3">
          <div className="p-4 bg-card border border-border rounded-lg">
            <h4 className="text-sm font-semibold mb-2 text-gold">Strengths</h4>
            <ul className="text-xs text-muted-foreground space-y-1">
              {kpi.totalScore >= 600 && <li>• High predicted team score</li>}
              {composition.allRounders >= 2 && <li>• Good mix of all-rounders for flexibility</li>}
              {composition.wicketKeepers >= 1 && <li>• Solid wicket-keeping presence</li>}
            </ul>
          </div>
          
          <div className="p-4 bg-card border border-border rounded-lg">
            <h4 className="text-sm font-semibold mb-2 text-destructive">Areas to Improve</h4>
            <ul className="text-xs text-muted-foreground space-y-1">
              {kpi.totalScore < 500 && <li>• Consider higher-scoring players</li>}
              {composition.wicketKeepers === 0 && <li>• Missing wicket-keeper in lineup</li>}
              {composition.allRounders === 0 && <li>• Team lacks versatility</li>}
            </ul>
          </div>

          <div className="p-4 bg-primary/10 border border-primary/20 rounded-lg">
            <h4 className="text-sm font-semibold mb-2">Match Strategy</h4>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {composition.batsmen >= 5 ? 'Batting-heavy lineup focused on high scores. ' : ''}
              {composition.bowlers >= 5 ? 'Bowling-focused team for defensive play. ' : ''}
              {composition.allRounders >= 3 ? 'Balanced approach with versatile all-rounders. ' : ''}
              Predicted total: {kpi.totalScore.toFixed(0)} points.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

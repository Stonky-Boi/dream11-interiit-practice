import { motion, AnimatePresence } from 'framer-motion';
import { Player } from '@/types/player';
import { X, ArrowLeft, ArrowRightLeft, TrendingUp, Award, Zap, Activity } from 'lucide-react';
import { Button } from './ui/button';

interface PlayerInspectPanelProps {
  player: Player | null;
  onClose: () => void;
  onSwap?: () => void;
}

export const PlayerInspectPanel = ({ player, onClose, onSwap }: PlayerInspectPanelProps) => {
  return (
    <AnimatePresence>
      {player && (
        <motion.div
          initial={{ x: -400, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: -400, opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="w-80 bg-card border-r border-border p-4 overflow-y-auto"
        >
          <div className="flex justify-between items-center mb-6">
            <Button variant="ghost" size="sm" onClick={onClose}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* Player Avatar */}
          <div className="flex flex-col items-center mb-6">
            <div className="relative mb-4">
              <div className="absolute inset-0 bg-gold rounded-full blur-xl opacity-30" />
              <img 
                src={player.avatar} 
                alt={player.name}
                className="relative w-32 h-32 rounded-full border-4 border-gold"
              />
              <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 bg-gold text-primary-foreground px-4 py-1 rounded-full font-bold">
                {player.predictedPoints.toFixed(1)}
              </div>
            </div>
            <h2 className="text-2xl font-bold text-center mb-1">{player.name}</h2>
            <p className="text-muted-foreground">{player.role}</p>
            <p className="text-sm text-muted-foreground">{player.team}</p>
          </div>

          {/* Stats */}
          <div className="space-y-4 mb-6">
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                <span className="font-medium">Predicted Points</span>
              </div>
              <span className="text-xl font-bold text-primary">{player.predictedPoints.toFixed(1)}</span>
            </div>
          </div>

          {/* Detailed Stats */}
          <div className="mb-6">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Career Stats
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Matches</span>
                <span className="font-medium">{player.stats.matches}</span>
              </div>
              {player.stats.runs !== undefined && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Runs</span>
                  <span className="font-medium">{player.stats.runs}</span>
                </div>
              )}
              {player.stats.wickets !== undefined && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Wickets</span>
                  <span className="font-medium">{player.stats.wickets}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-muted-foreground">Average</span>
                <span className="font-medium">{player.stats.average}</span>
              </div>
              {player.stats.strikeRate && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Strike Rate</span>
                  <span className="font-medium">{player.stats.strikeRate}</span>
                </div>
              )}
              {player.stats.economy && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Economy</span>
                  <span className="font-medium">{player.stats.economy}</span>
                </div>
              )}
            </div>
          </div>

          {/* SHAP Explanation */}
          <div className="mb-6 p-4 bg-primary/10 border border-primary/20 rounded-lg">
            <h3 className="font-semibold mb-2 text-sm">Why Selected?</h3>
            <p className="text-xs text-muted-foreground mb-3">
              Selected due to high consistency on this ground and excellent recent form.
            </p>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span>Runs contribution</span>
                <span className="text-secondary">+{player.shap.runs.toFixed(1)}</span>
              </div>
              <div className="flex justify-between">
                <span>Consistency boost</span>
                <span className="text-secondary">+{player.shap.consistency.toFixed(1)}</span>
              </div>
              <div className="flex justify-between">
                <span>Form impact</span>
                <span className="text-secondary">+{player.shap.form.toFixed(1)}</span>
              </div>
            </div>
          </div>

          {/* Recent Form Chart */}
          <div className="mb-6">
            <h3 className="font-semibold mb-3 text-sm">Last 5 Matches</h3>
            <div className="flex items-end gap-2 h-24">
              {player.recentForm.map((score, i) => (
                <div key={i} className="flex-1 flex flex-col items-center gap-1">
                  <span className="text-xs text-muted-foreground">{score}</span>
                  <div
                    className="w-full bg-gradient-to-t from-secondary to-secondary/50 rounded-t"
                    style={{ height: `${(score / 100) * 100}%` }}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Swap Button */}
          <Button 
            onClick={onSwap}
            className="w-full gold-gradient font-bold"
            size="lg"
          >
            <ArrowRightLeft className="w-4 h-4 mr-2" />
            Swap Player
          </Button>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

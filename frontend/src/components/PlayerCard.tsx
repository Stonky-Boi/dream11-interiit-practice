import { motion } from 'framer-motion';
import { Player } from '@/types/player';
import { TrendingUp, Award, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PlayerCardProps {
  player: Player;
  onClick?: () => void;
  selected?: boolean;
  compact?: boolean;
}

export const PlayerCard = ({ player, onClick, selected, compact }: PlayerCardProps) => {
  if (compact) {
    return (
      <motion.div
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onClick}
        className={cn(
          "relative flex flex-col items-center p-3 bg-card rounded-lg border-2 cursor-pointer transition-all",
          selected ? "border-gold glow-gold" : "border-border hover:border-gold/50"
        )}
      >
        <img src={player.avatar} alt={player.name} className="w-12 h-12 rounded-full mb-2" />
        <p className="text-xs font-medium text-center line-clamp-1">{player.name}</p>
        <p className="text-xs text-primary font-bold">{player.predictedPoints.toFixed(1)}</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -8, scale: 1.02 }}
      onClick={onClick}
      className={cn(
        "relative p-6 rounded-xl cursor-pointer transition-all card-hover",
        selected ? "bg-gradient-to-br from-gold/20 to-gold-dark/20 border-2 border-gold" : "bg-card border border-border"
      )}
      style={{ boxShadow: selected ? 'var(--shadow-gold)' : 'var(--shadow-card)' }}
    >
      {/* Gold accent corner */}
      <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-gold to-gold-dark opacity-20 rounded-bl-3xl" />
      
      <div className="flex items-start gap-4">
        <div className="relative">
          <img 
            src={player.avatar} 
            alt={player.name} 
            className="w-20 h-20 rounded-full border-2 border-gold"
          />
          <div className="absolute -bottom-1 -right-1 bg-gold text-primary-foreground rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold">
            {player.predictedPoints.toFixed(0)}
          </div>
        </div>

        <div className="flex-1">
          <h3 className="text-lg font-bold mb-1">{player.name}</h3>
          <p className="text-sm text-muted-foreground mb-2">{player.role} • {player.team}</p>
          
          <div className="flex gap-3 mb-3">
            <div className="flex items-center gap-1">
              <Award className="w-3 h-3 text-gold" />
              <span className="text-xs">C: {player.consistency}</span>
            </div>
            <div className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3 text-secondary" />
              <span className="text-xs">F: {player.form}</span>
            </div>
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3 text-primary" />
              <span className="text-xs">{player.predictedPoints.toFixed(1)}</span>
            </div>
          </div>

          {/* Mini sparkline */}
          <div className="flex items-end gap-1 h-8">
            {player.recentForm.map((score, i) => (
              <div
                key={i}
                className="flex-1 bg-gradient-to-t from-secondary to-secondary/50 rounded-t"
                style={{ height: `${(score / 100) * 100}%` }}
              />
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

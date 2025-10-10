import { motion } from 'framer-motion';
import { Player, PlayerRole } from '@/types/player';
import { PlayerCard } from './PlayerCard';
import { useState } from 'react';
import { Button } from './ui/button';

interface PlayerCarouselProps {
  players: Player[];
  onPlayerSelect: (player: Player) => void;
  selectedIds: string[];
}

export const PlayerCarousel = ({ players, onPlayerSelect, selectedIds }: PlayerCarouselProps) => {
  const [filter, setFilter] = useState<PlayerRole | 'All'>('All');

  const filteredPlayers = filter === 'All' 
    ? players 
    : players.filter(p => p.role === filter);

  const roles: (PlayerRole | 'All')[] = ['All', 'Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper'];

  return (
    <div className="h-64 border-t border-border bg-card/50 backdrop-blur-sm p-4">
      <div className="flex gap-2 mb-4">
        {roles.map((role) => (
          <Button
            key={role}
            variant={filter === role ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilter(role)}
            className={filter === role ? 'gold-gradient' : ''}
          >
            {role}
          </Button>
        ))}
      </div>

      <div className="flex gap-3 overflow-x-auto pb-4 scrollbar-thin scrollbar-thumb-gold scrollbar-track-muted">
        {filteredPlayers.map((player, index) => (
          <motion.div
            key={player.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="flex-shrink-0 w-28"
          >
            <PlayerCard
              player={player}
              onClick={() => onPlayerSelect(player)}
              selected={selectedIds.includes(player.id)}
              compact
            />
          </motion.div>
        ))}
      </div>
    </div>
  );
};

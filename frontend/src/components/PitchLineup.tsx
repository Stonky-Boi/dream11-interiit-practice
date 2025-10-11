import { motion } from 'framer-motion';
import { Player } from '@/types/player';
import { Plus } from 'lucide-react';

interface PitchLineupProps {
  lineup: (Player | null)[];
  onSlotClick: (index: number) => void;
}

export const PitchLineup = ({ lineup, onSlotClick }: PitchLineupProps) => {
  // Formation: 1 WK, 3-4 Batsmen, 2-3 All-rounders, 3-4 Bowlers
  const formation = [
    [0], // WK
    [1, 2, 3], // Batsmen top
    [4, 5], // Batsmen/All-rounders mid
    [6, 7, 8], // All-rounders/Bowlers
    [9, 10], // Bowlers
  ];

  return (
    <div className="flex-auto flex flex-col relative items-center p-0">
      <div className="w-3/4 max-w-xl pitch-texture rounded-2xl p-4 shadow-2xl">
        {/* Pitch lines */}
        <div className="relative w-full h-full">
          <div className="absolute inset-0 flex flex-col justify-around">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-px bg-foreground/10" />
            ))}
          </div>

          {/* Players in formation */}
          <div className="relative space-y-8">
            {formation.map((row, rowIndex) => (
              <div key={rowIndex} className="flex justify-center gap-6">
                {row.map((slotIndex) => {
                  const player = lineup[slotIndex];
                  return (
                    <motion.div
                      key={slotIndex}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => onSlotClick(slotIndex)}
                      className="cursor-pointer"
                    >
                      {player ? (
                        <div className="flex flex-col items-center">
                          <div className="relative">
                            <img
                              src={player.avatar}
                              alt={player.name}
                              className="w-16 h-16 rounded-full border-3 border-gold shadow-lg"
                            />
                            <div className="absolute -bottom-1 -right-1 bg-gold text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold">
                              {player.predictedPoints.toFixed(0)}
                            </div>
                          </div>
                          <p className="text-xs font-medium text-center mt-1 text-foreground max-w-[80px] line-clamp-1">
                            {player.name.split(' ').pop()}
                          </p>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center">
                          <div className="w-16 h-16 rounded-full border-2 border-dashed border-foreground/30 flex items-center justify-center bg-background/50 hover:bg-background/70 transition-colors">
                            <Plus className="w-6 h-6 text-muted-foreground" />
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">Empty</p>
                        </div>
                      )}
                    </motion.div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

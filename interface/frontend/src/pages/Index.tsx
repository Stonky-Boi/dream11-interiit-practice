import { useState, useEffect } from 'react';
import { Player, TeamKPI } from '@/types/player';
import { mockPlayers } from '@/data/mockPlayers';
import { PlayerInspectPanel } from '@/components/PlayerInspectPanel';
import { PitchLineup } from '@/components/PitchLineup';
import { PlayerCarousel } from '@/components/PlayerCarousel';
import { KPIPanel } from '@/components/KPIPanel';
import { Button } from '@/components/ui/button';
import { Sparkles, Play, Settings, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import { useToast } from '@/hooks/use-toast';
import { Link, useLocation } from 'react-router-dom';

const Index = () => {
  const { toast } = useToast();
  const location = useLocation();
  const { match = '', date = '', time = '', team1 = '', team2 = '' } = location.state || {};
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  const [lineup, setLineup] = useState<(Player | null)[]>(Array(11).fill(null));
  const [swappingIndex, setSwappingIndex] = useState<number | null>(null);
  
  // Get players from both teams
  const allMatchPlayers = mockPlayers;

  // Auto-select top 11 players on mount using ML prediction
  useEffect(() => {
    const top11 = [...allMatchPlayers]
      .sort((a, b) => b.predictedPoints - a.predictedPoints)
      .slice(0, 11);
    setLineup(top11);
    toast({
      title: "AI Prediction Complete! 🤖",
      description: "Best 11 players selected from both teams using ML model.",
    });
  }, []);

  const calculateKPI = (): TeamKPI => {
    const validPlayers = lineup.filter((p): p is Player => p !== null);
    // const consistency = validPlayers.reduce((sum, p) => sum + p.consistency, 0) / (validPlayers.length || 1);
    // const form = validPlayers.reduce((sum, p) => sum + p.form, 0) / (validPlayers.length || 1);
    const totalScore = validPlayers.reduce((sum, p) => sum + p.predictedPoints, 0);
    
    // Diversity: check role distribution
    const roles = new Set(validPlayers.map(p => p.role));
    const diversity = (roles.size / 4) * 100;

    return {
      // consistency,
      // diversity,
      // form,
      totalScore,
      deltas: {
        // consistency: 5.2,
        // diversity: 3.1,
        // form: -2.4,
        totalScore: 8.7,
      },
    };
  };

  const handleSlotClick = (index: number) => {
    if (swappingIndex !== null) {
      // Complete swap
      const newLineup = [...lineup];
      const temp = newLineup[swappingIndex];
      newLineup[swappingIndex] = newLineup[index];
      newLineup[index] = temp;
      setLineup(newLineup);
      setSwappingIndex(null);
      toast({
        title: "Players swapped!",
        description: "Your lineup has been updated.",
      });
    } else if (lineup[index]) {
      // Inspect player
      setSelectedPlayer(lineup[index]);
    } else {
      // Add player to empty slot
      setSwappingIndex(index);
    }
  };

  const handlePlayerSelect = (player: Player) => {
    if (swappingIndex !== null) {
      // Add to empty slot or replace
      const newLineup = [...lineup];
      newLineup[swappingIndex] = player;
      setLineup(newLineup);
      setSwappingIndex(null);
      toast({
        title: "Player added!",
        description: `${player.name} has been added to your lineup.`,
      });
    } else {
      // Just inspect
      setSelectedPlayer(player);
    }
  };

  const handleSwapClick = () => {
    if (selectedPlayer) {
      const index = lineup.findIndex(p => p?.id === selectedPlayer.id);
      if (index !== -1) {
        setSwappingIndex(index);
        setSelectedPlayer(null);
        toast({
          title: "Swap mode activated",
          description: "Select another player to swap with.",
        });
      }
    }
  };

  const handleOptimalTeam = () => {
    const optimal = allMatchPlayers
      .sort((a, b) => b.predictedPoints - a.predictedPoints)
      .slice(0, 11);
    setLineup(optimal);
    setSwappingIndex(null);
    setSelectedPlayer(null);
    toast({
      title: "Optimal Team Regenerated! ⚡",
      description: "AI has re-predicted the best 11 players from both teams.",
    });
  };

  const handleFinalize = () => {
    const validPlayers = lineup.filter((p): p is Player => p !== null);
    if (validPlayers.length !== 11) {
      toast({
        title: "Incomplete lineup",
        description: `You need 11 players. Currently: ${validPlayers.length}`,
        variant: "destructive",
      });
      return;
    }

    toast({
      title: "Team finalized! 🎉",
      description: `Total predicted points: ${calculateKPI().totalScore.toFixed(1)}`,
    });
  };

  return (
    <div className="min-h-screen flex flex-col relative">
      {/* Background Image */}
      <div 
        className="fixed inset-0 -z-10"
        style={{
          backgroundImage: 'url(public/bgimage.jpg)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-b from-background/80 via-background/70 to-background/80" />
      </div>

      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm bg-background/30">
        <div className="container mx-auto px-2 py-3 flex items-center justify-between gap-2">
          <motion.div 
            initial={{ x: -50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            className="flex items-center gap-2"
          >
            <Sparkles className="w-6 h-6 text-primary" />
            <div>
              <h1 className="text-xl font-bold">AI Best 11 Prediction</h1>
              {match && <p className="text-xs text-muted-foreground">{match} • {date} • {time}</p>}
            </div>
          </motion.div>

          <div className="flex gap-2">
            <Button 
              onClick={handleOptimalTeam}
              variant="outline"
              size="sm"
              className="font-bold border-primary text-primary hover:bg-primary/10"
            >
              <Zap className="w-4 h-4 mr-2" />
              Return Best 11
            </Button>
            {/* Link to Model UI */ }
            <Link to="/model">
              <Button variant="outline" size="sm">
                <Settings className="w-4 h-4 mr-2" />
                Model UI
              </Button>
            </Link>
            <Button onClick={handleFinalize} size="lg" className="bg-primary hover:bg-primary/90 font-bold">
              <Play className="w-4 h-4 mr-2" />
              Finalize Team
            </Button>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex-1 flex">
        {/* Left: Player Inspect Panel */}
        <PlayerInspectPanel
          player={selectedPlayer}
          onClose={() => setSelectedPlayer(null)}
          onSwap={handleSwapClick}
        />

        {/* Center: Pitch Lineup */}
        <PitchLineup lineup={lineup} onSlotClick={handleSlotClick} />

        {/* Right: KPI Panel */}
        <KPIPanel kpi={calculateKPI()} lineup={lineup} />
      </div>

      {/* Bottom: Player Carousel */}
      <PlayerCarousel
        players={allMatchPlayers.filter(
          (player) => !lineup.some((p) => p?.id === player.id)
        )}
        onPlayerSelect={handlePlayerSelect}
        selectedIds={lineup.filter((p): p is Player => p !== null).map(p => p.id)}
      />
    </div>
  );
};

export default Index;


// incaase needed for the future use: i/ps are date venure league ?? 

// import { useState, useEffect } from 'react';
// import { Player, TeamKPI } from '@/types/player';
// import { mockPlayers } from '@/data/mockPlayers';
// import { PlayerInspectPanel } from '@/components/PlayerInspectPanel';
// import { PitchLineup } from '@/components/PitchLineup';
// import { PlayerCarousel } from '@/components/PlayerCarousel';
// import { KPIPanel } from '@/components/KPIPanel';
// import { Button } from '@/components/ui/button';
// import { Sparkles, Play, Settings, Zap } from 'lucide-react';
// import { motion } from 'framer-motion';
// import { useToast } from '@/hooks/use-toast';
// import { Link, useLocation } from 'react-router-dom';

// const Index = () => {
//   const { toast } = useToast();
//   const location = useLocation();
//   const { selectedTeam = 'team1', match = '', date = '' } = location.state || {};
//   const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
//   const [lineup, setLineup] = useState<(Player | null)[]>(Array(11).fill(null));
//   const [swappingIndex, setSwappingIndex] = useState<number | null>(null);
  
//   const teamPlayers = mockPlayers.filter(p => p.team.toLowerCase().includes(selectedTeam === 'team1' ? 'india' : 'australia'));

//   // Auto-select top 11 players on mount
//   useEffect(() => {
//     const top11 = [...teamPlayers]
//       .sort((a, b) => b.predictedPoints - a.predictedPoints)
//       .slice(0, 11);
//     setLineup(top11);
//   }, [selectedTeam]);

//   const calculateKPI = (): TeamKPI => {
//     const validPlayers = lineup.filter((p): p is Player => p !== null);
//     const consistency = validPlayers.reduce((sum, p) => sum + p.consistency, 0) / (validPlayers.length || 1);
//     const form = validPlayers.reduce((sum, p) => sum + p.form, 0) / (validPlayers.length || 1);
//     const totalScore = validPlayers.reduce((sum, p) => sum + p.predictedPoints, 0);
    
//     // Diversity: check role distribution
//     const roles = new Set(validPlayers.map(p => p.role));
//     const diversity = (roles.size / 4) * 100;

//     return {
//       consistency,
//       diversity,
//       form,
//       totalScore,
//       deltas: {
//         consistency: 5.2,
//         diversity: 3.1,
//         form: -2.4,
//         totalScore: 8.7,
//       },
//     };
//   };

//   const handleSlotClick = (index: number) => {
//     if (swappingIndex !== null) {
//       // Complete swap
//       const newLineup = [...lineup];
//       const temp = newLineup[swappingIndex];
//       newLineup[swappingIndex] = newLineup[index];
//       newLineup[index] = temp;
//       setLineup(newLineup);
//       setSwappingIndex(null);
//       toast({
//         title: "Players swapped!",
//         description: "Your lineup has been updated.",
//       });
//     } else if (lineup[index]) {
//       // Inspect player
//       setSelectedPlayer(lineup[index]);
//     } else {
//       // Add player to empty slot
//       setSwappingIndex(index);
//     }
//   };

//   const handlePlayerSelect = (player: Player) => {
//     if (swappingIndex !== null) {
//       // Add to empty slot or replace
//       const newLineup = [...lineup];
//       newLineup[swappingIndex] = player;
//       setLineup(newLineup);
//       setSwappingIndex(null);
//       toast({
//         title: "Player added!",
//         description: `${player.name} has been added to your lineup.`,
//       });
//     } else {
//       // Just inspect
//       setSelectedPlayer(player);
//     }
//   };

//   const handleSwapClick = () => {
//     if (selectedPlayer) {
//       const index = lineup.findIndex(p => p?.id === selectedPlayer.id);
//       if (index !== -1) {
//         setSwappingIndex(index);
//         setSelectedPlayer(null);
//         toast({
//           title: "Swap mode activated",
//           description: "Select another player to swap with.",
//         });
//       }
//     }
//   };

//   const handleOptimalTeam = () => {
//     const optimal = teamPlayers
//       .sort((a, b) => b.predictedPoints - a.predictedPoints)
//       .slice(0, 11);
//     setLineup(optimal);
//     setSwappingIndex(null);
//     setSelectedPlayer(null);
//     toast({
//       title: "Optimal Team Generated! ⚡",
//       description: "AI has predicted the best 11 players for maximum points.",
//     });
//   };

//   const handleFinalize = () => {
//     const validPlayers = lineup.filter((p): p is Player => p !== null);
//     if (validPlayers.length !== 11) {
//       toast({
//         title: "Incomplete lineup",
//         description: `You need 11 players. Currently: ${validPlayers.length}`,
//         variant: "destructive",
//       });
//       return;
//     }

//     toast({
//       title: "Team finalized! 🎉",
//       description: `Total predicted points: ${calculateKPI().totalScore.toFixed(1)}`,
//     });
//   };

//   return (
//     <div className="min-h-screen stadium-gradient flex flex-col"
//     // style={{ backgroundImage: "url('public/bgimage.jpg')" }}
//     >
      
//       {/* Header */}
//       <header className="border-b border-border/50 backdrop-blur-sm bg-background/30">
//         <div className="container mx-auto px-4 py-4 flex items-center justify-between">
//           <motion.div 
//             initial={{ x: -50, opacity: 0 }}
//             animate={{ x: 0, opacity: 1 }}
//             className="flex items-center gap-3"
//           >
//             <Sparkles className="w-8 h-8 text-gold" />
//             <div>
//               <h1 className="text-2xl font-bold">Dream11 Team Builder</h1>
//               {match && <p className="text-xs text-muted-foreground">{match} • {date}</p>}
//             </div>
//           </motion.div>

//           <div className="flex gap-3">
//             <Button 
//               onClick={handleOptimalTeam}
//               variant="outline"
//               size="sm"
//               className="font-bold border-gold text-gold hover:bg-gold/10"
//             >
//               <Zap className="w-4 h-4 mr-2" />
//               Predict Optimal
//             </Button>
//             <Link to="/model">
//               <Button variant="outline" size="sm">
//                 <Settings className="w-4 h-4 mr-2" />
//                 Model UI
//               </Button>
//             </Link>
//             <Button onClick={handleFinalize} size="lg" className="gold-gradient font-bold">
//               <Play className="w-4 h-4 mr-2" />
//               Finalize Team
//             </Button>
//           </div>
//         </div>
//       </header>

//       {/* Main Layout */}
//       <div className="flex-1 flex">
//         {/* Left: Player Inspect Panel */}
//         <PlayerInspectPanel
//           player={selectedPlayer}
//           onClose={() => setSelectedPlayer(null)}
//           onSwap={handleSwapClick}
//         />

//         {/* Center: Pitch Lineup */}
//         <PitchLineup lineup={lineup} onSlotClick={handleSlotClick} />

//         {/* Right: KPI Panel */}
//         <KPIPanel kpi={calculateKPI()} lineup={lineup} />
//       </div>

//       {/* Bottom: Player Carousel */}
//       <PlayerCarousel
//         players={teamPlayers.filter(
//           (player) => !lineup.some((p) => p?.id === player.id)
//         )}
//         onPlayerSelect={handlePlayerSelect}
//         selectedIds={lineup.filter((p): p is Player => p !== null).map(p => p.id)}
//       />
//     </div>
//   );
// };

// export default Index;

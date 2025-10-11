// import { useState } from 'react';
// import { useNavigate } from 'react-router-dom';
// import { Button } from '@/components/ui/button';
// import { Label } from '@/components/ui/label';
// import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
// import { Calendar } from '@/components/ui/calendar';
// import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
// import { Sparkles, CalendarIcon, ArrowRight } from 'lucide-react';
// import { motion } from 'framer-motion';
// import { format } from 'date-fns';
// import { cn } from '@/lib/utils';

// const leagues = [
//   { id: 'ipl', name: 'Indian Premier League (IPL)' },
//   { id: 't20wc', name: 'T20 World Cup' },
//   { id: 'odi', name: 'ODI' },
//   { id: 'test', name: 'Test Match' },
// ];

// const matches = [
//   { id: '1', name: 'Mumbai Indians vs Chennai Super Kings', league: 'ipl', team1: 'Mumbai Indians', team2: 'Chennai Super Kings' },
//   { id: '2', name: 'Royal Challengers vs Kolkata Knight Riders', league: 'ipl', team1: 'Royal Challengers', team2: 'Kolkata Knight Riders' },
//   { id: '3', name: 'India vs Australia', league: 't20wc', team1: 'India', team2: 'Australia' },
//   { id: '4', name: 'England vs Pakistan', league: 't20wc', team1: 'England', team2: 'Pakistan' },
//   { id: '5', name: 'India vs South Africa', league: 'odi', team1: 'India', team2: 'South Africa' },
//   { id: '6', name: 'Australia vs England', league: 'test', team1: 'Australia', team2: 'England' },
// ];

// const MatchSelection = () => {
//   const navigate = useNavigate();
//   const [date, setDate] = useState<Date>();
//   const [selectedLeague, setSelectedLeague] = useState<string>('');
//   const [selectedMatch, setSelectedMatch] = useState<string>('');
//   const [selectedTeam, setSelectedTeam] = useState<string>('team1');

//   const filteredMatches = selectedLeague
//     ? matches.filter(m => m.league === selectedLeague)
//     : matches;

//   const currentMatch = matches.find(m => m.id === selectedMatch);

//   const handleContinue = () => {
//     if (date && selectedLeague && selectedMatch && selectedTeam) {
//       const match = matches.find(m => m.id === selectedMatch);
//       navigate('/team-builder', { 
//         state: { 
//           selectedTeam,
//           match: match?.name,
//           date: format(date, 'PPP')
//         } 
//       });
//     }
//   };

//   const isFormValid = date && selectedLeague && selectedMatch;

//   return (
//     <div className="min-h-screen stadium-gradient flex items-center justify-center p-4">
//       <motion.div
//         initial={{ opacity: 0, y: 20 }}
//         animate={{ opacity: 1, y: 0 }}
//         className="w-full max-w-2xl"
//       >
//         <div className="card-hover bg-background/10 backdrop-blur-xl border border-border/50 rounded-2xl p-8 shadow-2xl">
//           {/* Header */}
//           <div className="text-center mb-8">
//             <motion.div
//               initial={{ scale: 0.8 }}
//               animate={{ scale: 1 }}
//               className="flex items-center justify-center gap-3 mb-4"
//             >
//               <Sparkles className="w-10 h-10 text-gold animate-glow-pulse" />
//               <h1 className="text-4xl font-bold">Dream11 Predictor</h1>
//             </motion.div>
//             <p className="text-muted-foreground">Select match details to get AI-powered predictions</p>
//           </div>

//           {/* Form */}
//           <div className="space-y-6">
//             {/* Match Date */}
//             <div className="space-y-2">
//               <Label htmlFor="date" className="text-base">Match Date</Label>
//               <Popover>
//                 <PopoverTrigger asChild>
//                   <Button
//                     variant="outline"
//                     className={cn(
//                       "w-full justify-start text-left font-normal h-12",
//                       !date && "text-muted-foreground"
//                     )}
//                   >
//                     <CalendarIcon className="mr-2 h-4 w-4" />
//                     {date ? format(date, "PPP") : <span>Pick a date</span>}
//                   </Button>
//                 </PopoverTrigger>
//                 <PopoverContent className="w-auto p-0 bg-background border-border" align="start">
//                   <Calendar
//                     mode="single"
//                     selected={date}
//                     onSelect={setDate}
//                     initialFocus
//                     className="pointer-events-auto"
//                   />
//                 </PopoverContent>
//               </Popover>
//             </div>

//             {/* League Selection */}
//             <div className="space-y-2">
//               <Label htmlFor="league" className="text-base">Select League</Label>
//               <Select value={selectedLeague} onValueChange={setSelectedLeague}>
//                 <SelectTrigger className="w-full h-12 bg-background/50">
//                   <SelectValue placeholder="Choose a league" />
//                 </SelectTrigger>
//                 <SelectContent className="bg-background border-border">
//                   {leagues.map((league) => (
//                     <SelectItem key={league.id} value={league.id}>
//                       {league.name}
//                     </SelectItem>
//                   ))}
//                 </SelectContent>
//               </Select>
//             </div>

//             {/* Match Selection */}
//             <div className="space-y-2">
//               <Label htmlFor="match" className="text-base">Select Match</Label>
//               <Select 
//                 value={selectedMatch} 
//                 onValueChange={setSelectedMatch}
//                 disabled={!selectedLeague}
//               >
//                 <SelectTrigger className="w-full h-12 bg-background/50">
//                   <SelectValue placeholder={selectedLeague ? "Choose a match" : "Select a league first"} />
//                 </SelectTrigger>
//                 <SelectContent className="bg-background border-border">
//                   {filteredMatches.map((match) => (
//                     <SelectItem key={match.id} value={match.id}>
//                       {match.name}
//                     </SelectItem>
//                   ))}
//                 </SelectContent>
//               </Select>
//             </div>

//             {/* Team Selection */}
//             {selectedMatch && currentMatch && (
//               <div className="space-y-2">
//                 <Label className="text-base">Select Your Team</Label>
//                 <div className="grid grid-cols-2 gap-3">
//                   <Button
//                     variant={selectedTeam === 'team1' ? 'default' : 'outline'}
//                     className={`h-12 ${selectedTeam === 'team1' ? 'gold-gradient' : ''}`}
//                     onClick={() => setSelectedTeam('team1')}
//                   >
//                     {currentMatch.team1}
//                   </Button>
//                   <Button
//                     variant={selectedTeam === 'team2' ? 'default' : 'outline'}
//                     className={`h-12 ${selectedTeam === 'team2' ? 'gold-gradient' : ''}`}
//                     onClick={() => setSelectedTeam('team2')}
//                   >
//                     {currentMatch.team2}
//                   </Button>
//                 </div>
//               </div>
//             )}

//             {/* Continue Button */}
//             <Button
//               onClick={handleContinue}
//               disabled={!isFormValid}
//               className="w-full h-12 text-base gold-gradient font-bold"
//               size="lg"
//             >
//               Continue to Team Builder
//               <ArrowRight className="ml-2 h-5 w-5" />
//             </Button>
//           </div>
//         </div>
//       </motion.div>
//     </div>
//   );
// };

// export default MatchSelection;

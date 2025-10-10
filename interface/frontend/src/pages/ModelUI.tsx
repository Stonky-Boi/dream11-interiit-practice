import { useState, useEffect } from 'react';
import { ArrowLeft, Play, Download, BarChart3, TrendingUp } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { motion } from 'framer-motion';
import { useToast } from '@/hooks/use-toast';

const ModelUI = () => {
  const { toast } = useToast();
  const [trainStart, setTrainStart] = useState('2020-01-01');
  const [trainEnd, setTrainEnd] = useState('2024-06-30');
  const [testStart, setTestStart] = useState('2024-07-01');
  const [testEnd, setTestEnd] = useState('2024-12-31');
  const [selectedModel, setSelectedModel] = useState('xgboost');
  const [selectedDataset, setSelectedDataset] = useState('ODIs');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  const mockMetrics = {
    xgboost: { mae: 8.3, rmse: 12.1 },
    transformer: { mae: 7.9, rmse: 11.5},
    regression: { mae: 12.4, rmse: 16.8 },
  };

  useEffect(() => {
    if (isTraining && trainingProgress < 100) {
      const interval = setInterval(() => {
        setTrainingProgress(prev => {
          const increment = Math.random() * 8 + 2; // Random increment between 2-10%
          const newProgress = Math.min(prev + increment, 100);
          
          if (newProgress >= 100) {
            setIsTraining(false);
            toast({
              title: "Training completed!",
              description: "Model has been successfully trained and is ready for evaluation.",
              duration: 500,
            });
          }
          
          return newProgress;
        });
      }, 500);
      
      return () => clearInterval(interval);
    }
  }, [isTraining, trainingProgress, toast]);

  const handleTrain = () => {
    setIsTraining(true);
    setTrainingProgress(0);
    toast({
      title: "Training started",
      description: "Model training has been initiated. This may take several minutes.",
      duration: 500,
    });
  };

  const handleEvaluate = () => {
    toast({
      title: "Evaluation complete",
      description: `MAE: ${mockMetrics[selectedModel as keyof typeof mockMetrics].mae} 
      | RMSE: ${mockMetrics[selectedModel as keyof typeof mockMetrics].rmse}`,
    });
  };

  const handleExport = () => {
    toast({
      title: "Exporting data",
      description: "CSV file will be downloaded shortly.",
      duration: 500,
    });
  };

  return (
    <div className="min-h-screen stadium-gradient">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm bg-background/30">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Product UI
              </Button>
            </Link>
            <div>
              <h1 className="text-2xl font-bold">Model Developer UI</h1>
              <p className="text-xs text-muted-foreground">Training, Evaluation & Explainability</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <Tabs defaultValue="train" className="space-y-6">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="train">Train</TabsTrigger>
            <TabsTrigger value="evaluate">Evaluate</TabsTrigger>
            <TabsTrigger value="compare">Compare</TabsTrigger>
          </TabsList>

          {/* Training Tab */}
          <TabsContent value="train" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle>Model Training</CardTitle>
                  <CardDescription>
                    Configure training parameters and retrain models on historical data
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="train-start">Training Start Date</Label>
                      <Input
                        id="train-start"
                        type="date"
                        value={trainStart}
                        onChange={(e) => setTrainStart(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="train-end">Training End Date</Label>
                      <Input
                        id="train-end"
                        type="date"
                        value={trainEnd}
                        max="2024-06-30"
                        onChange={(e) => setTrainEnd(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="model-type">Model Type</Label>
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger id="model-type">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="xgboost">XGBoost</SelectItem>
                          <SelectItem value="transformer">Transformer (LSTM)</SelectItem>
                          <SelectItem value="regression">Linear Regression</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="dataset-type">Dataset</Label>
                      <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                        <SelectTrigger id="dataset-type">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="ODIs">ODI Matches</SelectItem>
                          <SelectItem value="T-20 Matches">T20 Matches</SelectItem>
                          <SelectItem value="Test Matches">Test Matches</SelectItem>
                          <SelectItem value="All Matches">All Matches</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm text-muted-foreground mb-2">
                      <strong>Note:</strong> Training data cutoff is 2024-06-30. No data after this date will be used for training.
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Available datasets: ODI ({selectedDataset === 'odi' ? '✓ Selected' : 'odi.xlsx'}), 
                      T20 ({selectedDataset === 't20' ? '✓ Selected' : 't20.xlsx'}), 
                      Test ({selectedDataset === 'test' ? '✓ Selected' : 'testdata.xlsx'})
                    </p>
                  </div>

                  {isTraining && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-3 p-4 bg-primary/10 border border-primary/20 rounded-lg"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold">Training Progress</span>
                        <span className="text-2xl font-bold text-gold">{Math.round(trainingProgress)}%</span>
                      </div>
                      <Progress value={trainingProgress} className="h-3" />
                      <p className="text-xs text-muted-foreground">
                        {trainingProgress < 30 && "Loading dataset and preprocessing..."}
                        {trainingProgress >= 30 && trainingProgress < 60 && "Training model on historical data..."}
                        {trainingProgress >= 60 && trainingProgress < 90 && "Optimizing hyperparameters..."}
                        {trainingProgress >= 90 && trainingProgress < 100 && "Finalizing and validating model..."}
                      </p>
                    </motion.div>
                  )}

                  <Button 
                    onClick={handleTrain} 
                    className="w-full gold-gradient font-bold" 
                    size="lg"
                    disabled={isTraining}
                  >
                    <Play className="w-4 h-4 mr-2" />
                    {isTraining ? 'Training in Progress...' : 'Start Training'}
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Evaluation Tab */}
          <TabsContent value="evaluate" className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Evaluation Configuration</CardTitle>
                    <CardDescription>Set test data range for model evaluation</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="test-start">Test Start Date</Label>
                      <Input
                        id="test-start"
                        type="date"
                        value={testStart}
                        onChange={(e) => setTestStart(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="test-end">Test End Date</Label>
                      <Input
                        id="test-end"
                        type="date"
                        value={testEnd}
                        onChange={(e) => setTestEnd(e.target.value)}
                      />
                    </div>
                    <Button onClick={handleEvaluate} className="w-full" size="lg">
                      <BarChart3 className="w-4 h-4 mr-2" />
                      Run Evaluation
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Performance Metrics</CardTitle>
                    <CardDescription>Current model: {selectedModel.toUpperCase()}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-secondary/20 rounded-lg">
                        <p className="text-xs text-muted-foreground mb-1">MAE</p>
                        <p className="text-2xl font-bold">
                          {mockMetrics[selectedModel as keyof typeof mockMetrics].mae}
                        </p>
                      </div>
                      <div className="text-center p-4 bg-secondary/20 rounded-lg">
                        <p className="text-xs text-muted-foreground mb-1">RMSE</p>
                        <p className="text-2xl font-bold">
                          {mockMetrics[selectedModel as keyof typeof mockMetrics].rmse}
                        </p>
                      </div>
                      {/* <div className="text-center p-4 bg-gold/20 rounded-lg">
                        <p className="text-xs text-muted-foreground mb-1">R²</p>
                        <p className="text-2xl font-bold">
                          {mockMetrics[selectedModel as keyof typeof mockMetrics].r2}
                        </p>
                      </div> */}
                    </div>

                    <Button onClick={handleExport} variant="outline" className="w-full">
                      <Download className="w-4 h-4 mr-2" />
                      Export Results (CSV)
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Predicted vs Actual Chart Placeholder */}
            <Card>
              <CardHeader>
                <CardTitle>Predicted vs Actual Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center bg-muted/50 rounded-lg">
                  <p className="text-muted-foreground">Chart visualization placeholder</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Compare Tab */}
          <TabsContent value="compare" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle>Model Comparison</CardTitle>
                  <CardDescription>Side-by-side performance metrics</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left p-3 font-semibold">Model</th>
                          <th className="text-center p-3 font-semibold">MAE</th>
                          <th className="text-center p-3 font-semibold">RMSE</th>
                          <th className="text-center p-3 font-semibold">R²</th>
                          <th className="text-center p-3 font-semibold">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(mockMetrics).map(([model, metrics]) => (
                          <tr key={model} className="border-b border-border/50">
                            <td className="p-3 font-medium">{model.toUpperCase()}</td>
                            <td className="text-center p-3">{metrics.mae}</td>
                            <td className="text-center p-3">{metrics.rmse}</td>
                            {/* <td className="text-center p-3 font-bold text-gold">{metrics.r2}</td> */}
                            {/* <td className="text-center p-3">
                              {metrics.r2 > 0.85 ? (
                                <span className="px-2 py-1 bg-secondary/20 text-secondary text-xs rounded-full">
                                  Excellent
                                </span>
                              ) : (
                                <span className="px-2 py-1 bg-muted text-muted-foreground text-xs rounded-full">
                                  Good
                                </span>
                              )}
                            </td> */}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              {/* SHAP Feature Importance */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-gold" />
                    Feature Importance (SHAP Values)
                  </CardTitle>
                  <CardDescription>Top contributing features for predictions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {[
                      { feature: 'Recent Form (Last 5)', importance: 0.28 },
                      { feature: 'Career Average', importance: 0.22 },
                      { feature: 'Venue Performance', importance: 0.18 },
                      { feature: 'Opposition Strength', importance: 0.15 },
                      { feature: 'Player Consistency', importance: 0.17 },
                    ].map((item, i) => (
                      <div key={i}>
                        <div className="flex justify-between text-sm mb-1">
                          <span>{item.feature}</span>
                          <span className="font-bold">{(item.importance * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${item.importance * 100}%` }}
                            transition={{ duration: 0.5, delay: i * 0.1 }}
                            className="h-full bg-gradient-to-r from-gold to-secondary"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default ModelUI;

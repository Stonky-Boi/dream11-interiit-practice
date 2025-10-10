const express = require('express');
const cors = require('cors');
const { scrapeMatchSquads } = require('./scraper'); 

const app = express();
const PORT = 3001; 

app.use(cors()); 

app.get('/api/squads', async (req, res) => {
  const matchUrl = req.query.url;

  if (!matchUrl) {
    return res.status(400).json({ error: 'Match URL is required' });
  }

  try {
    const players = await scrapeMatchSquads(matchUrl);
    res.json(players);
  } catch (error) {
    console.error('Scraping failed:', error);
    res.status(500).json({ error: 'Failed to scrape squad data' });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
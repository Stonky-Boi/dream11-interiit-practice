// scraper.js
const puppeteer = require('puppeteer');

async function scrapeMatchSquads(url) {
  const browser = await puppeteer.launch({ headless: "new" }); // Use "new" headless mode
  const page = await browser.newPage();

  try {
    console.log(`Navigating to ${url}...`);
    await page.goto(url, { waitUntil: 'networkidle2' });

    // Wait for the specific section containing squads to be visible
    await page.waitForSelector('.cb-min-inf', { timeout: 10000 });
    console.log('Squad section found.');

    const squads = await page.evaluate(() => {
      // This function runs in the browser's context
      const players = [];

      // Select the two team containers. This selector may need updating.
      const teamContainers = document.querySelectorAll('.cb-min-inf.cb-col-100 > .cb-col');

      if (teamContainers.length < 2) {
         console.error("Could not find two team containers.");
         return [];
      }

      const team1Name = teamContainers[0].querySelector('.cb-col-100.cb-min-tm-nm').innerText.trim();
      const team2Name = teamContainers[1].querySelector('.cb-col-100.cb-min-tm-nm').innerText.trim();

      // Scrape players from the first team
      const team1Players = teamContainers[0].querySelectorAll('.cb-col-50 a.cb-text-link');
      team1Players.forEach(el => {
        const name = el.innerText.trim();
        // Cricbuzz often adds "(c)" or "(wk)" in the name, we can try to parse it
        let role = 'Player';
        if (name.includes('(c)')) role = 'Captain';
        if (name.includes('(wk)')) role = 'Wicket-Keeper';

        players.push({ name: name.replace(/\s*\(.*\)\s*/, ''), team: team1Name, role });
      });

      // Scrape players from the second team
      const team2Players = teamContainers[1].querySelectorAll('.cb-col-50 a.cb-text-link');
      team2Players.forEach(el => {
        const name = el.innerText.trim();
        let role = 'Player';
        if (name.includes('(c)')) role = 'Captain';
        if (name.includes('(wk)')) role = 'Wicket-Keeper';

        players.push({ name: name.replace(/\s*\(.*\)\s*/, ''), team: team2Name, role });
      });

      return players;
    });

    console.log(`Scraped ${squads.length} players.`);
    return squads;

  } catch (error) {
    console.error("An error occurred during scraping:", error);
    throw error; 
  } finally {
    await browser.close();
  }
}

module.exports = { scrapeMatchSquads };
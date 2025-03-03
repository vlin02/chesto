import WebSocket from 'ws';
import { URLSearchParams } from 'url';

// Get username and password from command line arguments
const username = process.argv[2];
const password = process.argv[3];

if (!username || !password) {
  console.log('Usage: node login.js USERNAME PASSWORD');
  process.exit(1);
}

// Connect to Pokémon Showdown server
const ws = new WebSocket('wss://sim3.psim.us/showdown/websocket');

ws.on('open', () => {
  console.log('Connected to server');
});

ws.on('message', async (data) => {
  const message = data.toString();
  
  // Parse the challstr
  if (message.includes('|challstr|')) {
    const challstr = message.split('|challstr|')[1];
    console.log('Received challstr, logging in...');
    
    try {
      // Make login request
      const loginUrl = 'https://play.pokemonshowdown.com/api/login';
      const params = new URLSearchParams();
      params.append('name', username);
      params.append('pass', password);
      params.append('challstr', challstr);
      
      const response = await fetch(loginUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: params.toString()
      });
      
      const responseText = await response.text();
      // Parse response (which starts with '])
      const jsonResponse = JSON.parse(responseText.substring(1));
      const assertion = jsonResponse.assertion;
      
      // Send the trn command to complete login
      ws.send(`|/trn ${username},0,${assertion}`);
      console.log('Login credentials sent');
    } catch (error) {
      console.error('Login error:', error.message);
    }
  }
  
  // Check for successful login
  if (message.includes('|updateuser|')) {
    const parts = message.split('|');
    const loggedInName = parts[2];
    const isGuest = parts[3] === '0';
    
    if (!isGuest && loggedInName.toLowerCase().startsWith(username.toLowerCase())) {
      console.log(`Successfully logged in as ${loggedInName}`);
      // Now you can send commands like joining rooms, sending messages, etc.
      // Example: ws.send('|/join lobby');
    }
  }
});

ws.on('error', (error) => {
  console.error('WebSocket error:', error.message);
});

ws.on('close', () => {
  console.log('Disconnected from server');
});
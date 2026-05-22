# Parlay Analyzer with OpenRouter AI

## Overview
This guide shows how to add AI-powered parlay analysis to your HR Oracle app using OpenRouter.

## ⚠️ IMPORTANT SECURITY NOTE
**NEVER** commit your API key directly to GitHub or embed it in client-side JavaScript! Anyone can view your HTML source and steal your key.

## Option 1: Secure Cloudflare Worker Proxy (RECOMMENDED)

### Step 1: Create a Cloudflare Worker
1. Go to [workers.cloudflare.com](https://workers.cloudflare.com) (free tier available)
2. Click "Create a Service"
3. Name it `openrouter-proxy`
4. Paste this code:

```javascript
export default {
  async fetch(request, env) {
    // Only allow requests from your GitHub Pages domain
    const allowedOrigin = 'https://bndrwe.github.io';
    const origin = request.headers.get('Origin');
    
    if (origin !== allowedOrigin) {
      return new Response('Forbidden', { status: 403 });
    }

    const body = await request.json();
    
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': allowedOrigin,
        'X-Title': 'HR Oracle Parlay Analyzer'
      },
      body: JSON.stringify(body)
    });

    const data = await response.json();
    
    return new Response(JSON.stringify(data), {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': allowedOrigin,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
      }
    });
  }
};
```

5. In Cloudflare Workers dashboard:
   - Go to Settings > Variables
   - Add an Environment Variable:
     - Name: `OPENROUTER_API_KEY`
     - Value: Your OpenRouter API key (starts with sk-or-v1-)
     - Click "Encrypt" to keep it secret
6. Save and deploy
7. Copy your worker URL (will be like `https://openrouter-proxy.YOUR_SUBDOMAIN.workers.dev`)

### Step 2: Add to index.html

Add this HTML right before the `</body>` tag:

```html
<!-- Parlay Analyzer -->
<div id="parlayAnalyzer" style="position:fixed;bottom:20px;right:20px;width:350px;background:var(--bg);border:2px solid var(--border);border-radius:8px;padding:15px;box-shadow:0 4px 12px rgba(0,0,0,0.3);z-index:1000;display:none">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <h3 style="margin:0;color:var(--amber)">🎯 AI Parlay Analyzer</h3>
    <button onclick="document.getElementById('parlayAnalyzer').style.display='none'" style="background:none;border:none;color:var(--text);font-size:20px;cursor:pointer">×</button>
  </div>
  <textarea id="parlayInput" placeholder="Paste your parlay picks here..." style="width:100%;height:100px;background:var(--bg2);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:8px;font-family:inherit;resize:vertical"></textarea>
  <button onclick="analyzeParlay()" style="width:100%;margin-top:10px;padding:10px;background:var(--amber);color:black;border:none;border-radius:4px;font-weight:bold;cursor:pointer">Analyze Parlay</button>
  <div id="parlayResult" style="margin-top:10px;color:var(--text);font-size:14px"></div>
</div>
<button onclick="document.getElementById('parlayAnalyzer').style.display='block'" style="position:fixed;bottom:20px;right:20px;width:50px;height:50px;border-radius:50%;background:var(--amber);border:none;font-size:24px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.3)">🤖</button>
```

### Step 3: Add JavaScript Function

Add this script in your `<head>` section:

```javascript
const PROXY_URL = 'https://openrouter-proxy.YOUR_SUBDOMAIN.workers.dev'; // Replace with your worker URL

async function analyzeParlay() {
  const input = document.getElementById('parlayInput').value;
  const resultDiv = document.getElementById('parlayResult');
  
  if (!input.trim()) {
    resultDiv.innerHTML = '<span style="color:red">Please enter parlay picks!</span>';
    return;
  }
  
  resultDiv.innerHTML = '<span style="color:var(--amber)">Analyzing...</span>';
  
  try {
    const response = await fetch(PROXY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'openai/gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'You are a baseball analytics expert. Analyze parlay bets and provide insights on probability, risk, and value.'
          },
          {
            role: 'user',
            content: `Analyze this parlay: ${input}`
          }
        ]
      })
    });
    
    const data = await response.json();
    const analysis = data.choices[0].message.content;
    
    resultDiv.innerHTML = `<div style="padding:10px;background:var(--bg2);border-radius:4px">${analysis}</div>`;
  } catch (error) {
    resultDiv.innerHTML = `<span style="color:red">Error: ${error.message}</span>`;
  }
}
```

## Security Benefits

✅ API key never exposed in browser  
✅ Only your domain can use the proxy  
✅ Free to host on Cloudflare Workers  
✅ Easy to update key without redeploying site  

## Next Steps

1. Deploy the Cloudflare Worker with your API key stored securely
2. Update index.html with the parlay analyzer UI
3. Replace PROXY_URL with your actual worker URL
4. Push to GitHub Pages
5. Test the parlay analyzer on your live site

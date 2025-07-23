# 🌍 AI Travel Planning Assistant

An intelligent travel planning application powered by AI agents that creates personalized itineraries, researches destinations, and provides budget breakdowns.

## ✨ Features

- **🤖 AI-Powered City Selection**: Get recommendations based on your travel preferences
- **🗺️ Destination Research**: Comprehensive insights about attractions, cuisine, and culture  
- **📅 Detailed Itineraries**: Day-by-day travel plans with activities and restaurants
- **💸 Budget Planning**: Realistic cost breakdowns for accommodations, meals, and activities
- **🎯 Customizable Preferences**: Choose travel type, interests, season, and duration

## 🚀 Live Demo

[Deploy on Streamlit Cloud](https://share.streamlit.io/)

## 🛠️ Local Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ai-travel-agent
```

2. **Install dependencies**
```bash
pip install -r requirements_deploy.txt
```

**Note**: This project uses compatible versions to avoid ChromaDB/Python 3.13 issues on cloud platforms.

3. **Set up environment variables**
Create a `.env.local` file with:
```
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

4. **Run the application**
```bash
streamlit run main.py
```

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repo
4. Add secrets in the Streamlit Cloud dashboard

### Option 2: Railway
1. Connect GitHub repo to Railway
2. Add environment variables
3. Deploy with one click

### Option 3: Render
1. Connect GitHub repo to Render
2. Set build command: `pip install -r requirements_deploy.txt`
3. Set start command: `streamlit run main.py --server.port $PORT`

## 🔑 Required API Keys

- **Groq API**: Get free API key from [console.groq.com](https://console.groq.com/)
- **HuggingFace API**: Get free API key from [huggingface.co](https://huggingface.co/)

## 🎯 Usage

1. Select your travel preferences in the sidebar
2. Click "🚀 Generate Travel Plan"  
3. View results in organized tabs:
   - **Cities**: Recommended destinations
   - **Research**: Detailed destination insights
   - **Itinerary**: Day-by-day travel plan
   - **Budget**: Cost breakdown
   - **Debug**: Technical data (for developers)

## 🤖 AI Agents

The application uses specialized AI agents:
- **City Selection Expert**: Recommends destinations
- **Local Destination Expert**: Provides detailed research
- **Professional Travel Planner**: Creates itineraries  
- **Travel Budget Specialist**: Calculates costs

## 📝 License

MIT License - feel free to use and modify!

## 🐛 Issues & Support

Create an issue on GitHub for bug reports or feature requests.

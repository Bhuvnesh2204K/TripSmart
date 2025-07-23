import streamlit as st
from groq import Groq
import os
import re
from dotenv import load_dotenv

# Load environment variables
# Try Streamlit secrets first (for deployment), then fall back to .env.local
try:
    import streamlit as st
    # Use Streamlit secrets if available (deployment)
    if hasattr(st, 'secrets') and st.secrets:
        for key, value in st.secrets.items():
            os.environ[key] = str(value)
except:
    pass

# Fallback to .env.local for local development
load_dotenv(dotenv_path=".env.local")

# Function to clean invalid surrogate characters (Windows terminal issue)
def clean_surrogates(text):
    if isinstance(text, str):
        return re.sub(r'[\ud800-\udfff]', '', text)
    return text

def get_groq_response(prompt, client):
    """Get response from Groq API"""
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_city_recommendations(inputs, client):
    """Generate city recommendations using Groq"""
    prompt = f"""
    As a travel expert, recommend 3 best cities to visit based on these preferences:
    
    Travel Type: {inputs['travel_type']}
    Interests: {inputs['interests']}
    Season: {inputs['season']}
    Budget Range: {inputs['budget']}
    Duration: {inputs['duration']} days
    
    For each city, provide:
    1. City name and country
    2. Why it matches their travel type and interests
    3. Why it's perfect for the specified season
    4. Brief highlight of what makes it special
    
    Format as a numbered list with city names clearly stated.
    """
    return get_groq_response(prompt, client)

def generate_destination_research(city, inputs, client):
    """Generate detailed destination research"""
    prompt = f"""
    Provide comprehensive research about {city} for a traveler with these preferences:
    Travel Type: {inputs['travel_type']}
    Interests: {inputs['interests']}
    Duration: {inputs['duration']} days
    
    Include:
    1. TOP 5 ATTRACTIONS: Must-visit places with brief descriptions and estimated time
    2. LOCAL CUISINE: Signature dishes and where to find them
    3. CULTURAL INSIGHTS: Important customs, etiquette, and cultural norms
    4. ACCOMMODATION: Best areas to stay for different budgets
    5. TRANSPORTATION: How to get around the city efficiently
    6. BEST TIME TO VISIT: Optimal months, weather, and crowds
    7. PRACTICAL TIPS: Currency, safety, language basics, common scams
    
    Make this practical and actionable for travelers.
    """
    return get_groq_response(prompt, client)

def generate_itinerary(city, inputs, client):
    """Generate detailed itinerary"""
    prompt = f"""
    Create a detailed {inputs['duration']}-day itinerary for {city} based on:
    - Travel Type: {inputs['travel_type']}
    - Interests: {inputs['interests']}
    - Duration: {inputs['duration']} days
    
    For each day, include:
    1. DAY NUMBER: Clearly state the day (e.g., 'Day 1: Arrival and City Exploration')
    2. Morning activities (9 AM - 12 PM) with specific attractions/locations
    3. Afternoon activities (12 PM - 5 PM) with specific attractions/locations
    4. Evening activities (5 PM - 9 PM) including dinner recommendations
    5. Recommended restaurants for each meal with cuisine type
    6. Transportation methods between locations
    7. Estimated time for each activity
    8. Booking requirements or tips
    
    Make activities geographically logical and account for travel time.
    """
    return get_groq_response(prompt, client)

def generate_budget_plan(city, inputs, client):
    """Generate budget breakdown"""
    prompt = f"""
    Create a realistic budget plan for a {inputs['duration']}-day trip to {city} 
    with a {inputs['budget']} budget.
    
    Break down costs for:
    1. ACCOMMODATION: Per night costs and total
    2. MEALS: Daily averages for Breakfast, Lunch, Dinner
    3. TRANSPORTATION: Local transport, airport transfers
    4. ATTRACTIONS: Entry fees for major sights
    5. SHOPPING & MISCELLANEOUS: Daily allowance for souvenirs, snacks
    6. EMERGENCY FUND: 10-15% buffer of total cost
    
    Provide:
    - Daily budget estimates for each category
    - Total estimated budget for the entire trip
    - Money-saving tips specific to {city}
    - Splurge recommendations with costs
    
    Use local currency or USD with clear indication.
    """
    return get_groq_response(prompt, client)

def main():
    st.set_page_config(page_title="AI Travel Planner 🧳", page_icon="🌍", layout="wide")
    st.title("🌍 AI Travel Planning Assistant")

    st.markdown("""
    Welcome to your AI-powered travel planner! Tell us your preferences, and our AI
    will craft a personalized itinerary and budget for your next adventure.
    """)

    with st.sidebar:
        st.header("⚙️ Trip Preferences")
        travel_type = st.selectbox("✈️ Travel Type", ["Leisure", "Business", "Adventure", "Cultural", "Relaxation", "Family Trip"], index=0)
        interests = st.multiselect("🎯 Interests", ["History", "Food", "Nature", "Art", "Shopping", "Nightlife", "Beaches", "Mountains", "Museums", "Sports", "Music", "Wildlife"])
        season = st.selectbox("🌤️ Season", ["Summer", "Winter", "Spring", "Fall", "Any"], index=4)
        duration = st.slider("🕐 Trip Duration (days)", 1, 14, 7)
        budget = st.selectbox("💰 Budget Range (per day, excluding flights)",
                              ["Budget (Rs500-Rs1500)", "Mid-range (Rs1500-Rs4000)", "Luxury (Rs4000+)"], index=1)

    if st.button("🚀 Generate Travel Plan", key="generate_button", help="Click to start the AI planning process"):
        if not interests:
            st.warning("⚠️ Please select at least one interest to help us plan your trip better.")
            return

        # Check if Groq API key is available
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("❌ Groq API key not found. Please check your configuration.")
            return

        inputs = {
            "travel_type": travel_type,
            "interests": ', '.join(interests),
            "season": season,
            "duration": duration,
            "budget": budget
        }

        # Initialize Groq client
        client = Groq(api_key=groq_api_key)

        with st.spinner("🧠 AI is working on your perfect trip... This may take a few minutes. Please be patient."):
            try:
                # Generate city recommendations
                st.subheader("🗺️ Your AI-Generated Travel Plan")
                
                progress_bar = st.progress(0)
                
                # Step 1: City Selection
                with st.status("🌆 Selecting perfect cities for you..."):
                    city_recommendations = generate_city_recommendations(inputs, client)
                    progress_bar.progress(25)
                
                # Extract first city for detailed planning (simple extraction)
                lines = city_recommendations.split('\n')
                selected_city = "Paris"  # Default fallback
                for line in lines:
                    if any(char.isdigit() for char in line) and any(char.isalpha() for char in line):
                        # Try to extract city name from numbered list
                        parts = line.split('.')
                        if len(parts) > 1:
                            city_part = parts[1].split(',')[0].split('-')[0].strip()
                            if len(city_part) > 2:
                                selected_city = city_part
                                break

                # Use tabs for better organization
                tab1, tab2, tab3, tab4 = st.tabs(["Cities", "Research", "Itinerary", "Budget"])

                with tab1:
                    st.header("🌆 Recommended Cities")
                    st.markdown(f'<div style="font-size: 16px;">{clean_surrogates(city_recommendations)}</div>', unsafe_allow_html=True)

                # Step 2: Research
                with st.status(f"🗺️ Researching {selected_city}..."):
                    research = generate_destination_research(selected_city, inputs, client)
                    progress_bar.progress(50)

                with tab2:
                    st.header("🗺 Destination Insights")
                    st.markdown(f'<div style="font-size: 16px;">{clean_surrogates(research)}</div>', unsafe_allow_html=True)

                # Step 3: Itinerary
                with st.status(f"📅 Creating {duration}-day itinerary..."):
                    itinerary = generate_itinerary(selected_city, inputs, client)
                    progress_bar.progress(75)

                with tab3:
                    st.header("📅 Detailed Itinerary")
                    st.markdown(f'<div style="font-size: 16px;">{clean_surrogates(itinerary)}</div>', unsafe_allow_html=True)

                # Step 4: Budget
                with st.status("💸 Calculating budget breakdown..."):
                    try:
                        budget_plan = generate_budget_plan(selected_city, inputs, client)
                        progress_bar.progress(100)
                        
                        with tab4:
                            st.header("💸 Budget Breakdown")
                            st.markdown(f'<div style="font-size: 16px;">{clean_surrogates(budget_plan)}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        with tab4:
                            st.header("💸 Budget Breakdown")
                            st.warning("⚠️ Budget calculation hit rate limits. Please try again in a few minutes.")
                            st.info("This is normal for free API tiers. The core travel planning is complete!")

                # Success message at the very end
                st.success("✅ Trip planning completed! Enjoy your journey! 🎉")

            except Exception as e:
                st.error("❌ An error occurred while generating the travel plan:")
                st.exception(e)
                st.info("""
                **Possible solutions:**
                1. Wait 1-2 minutes and try again (API rate limits)
                2. Try a shorter trip duration (fewer API calls)
                3. Select fewer interests
                4. Check if your Groq API key is valid
                """)

if __name__ == "__main__":
    main()

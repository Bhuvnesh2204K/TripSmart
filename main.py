import streamlit as st
from trip_agents import TripCrew
from dotenv import load_dotenv
import os
import re

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

def main():
    st.set_page_config(page_title="AI Travel Planner 🧳", page_icon="🌍", layout="wide")
    st.title("🌍 AI Travel Planning Assistant")

    st.markdown("""
    Welcome to your AI-powered travel planner! Tell us your preferences, and our team of AI agents
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

        inputs = {
            "travel_type": travel_type,
            "interests": ', '.join(interests),
            "season": season,
            "duration": duration,
            "budget": budget
        }

        with st.spinner("🧠 AI Agents are working on your perfect trip... This may take a few minutes. Please be patient."):
            try:
                crew_output = TripCrew(inputs).run()

                st.subheader("🗺️ Your AI-Generated Travel Plan")

                # Use tabs for better organization
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cities", "Research", "Itinerary", "Budget", "Debug"])

                with tab1:
                    st.header("🌆 Recommended Cities")
                    st.markdown(f'<div style="font-size: 16px;">{clean_surrogates(crew_output.get("🌆 City Selection", "🚫 No city selection found."))}</div>', unsafe_allow_html=True)

                with tab2:
                    st.header("🗺 Destination Insights")
                    st.markdown(f'<div style="font-size: 16px;">{clean_surrogates(crew_output.get("🗺 City Research", "🚫 No city research found."))}</div>', unsafe_allow_html=True)

                with tab3:
                    st.header("📅 Detailed Itinerary")
                    st.markdown(f'<div style="font-size: 16px;">{clean_surrogates(crew_output.get("📅 Itinerary", "🚫 No itinerary generated."))}</div>', unsafe_allow_html=True)

                with tab4:
                    st.header("💸 Budget Breakdown")
                    st.markdown(f'<div style="font-size: 16px;">{clean_surrogates(crew_output.get("💸 Budget Plan", "🚫 No budget breakdown available."))}</div>', unsafe_allow_html=True)

                with tab5:
                    st.header("🧪 Debug Output")
                    st.info("🔧 This section shows the raw technical data from the AI agents. It's mainly useful for developers to troubleshoot issues.")
                    with st.expander("View Raw Data (Technical)"):
                        st.json(crew_output)
                
                # Success message at the very end
                st.success("✅ Trip planning completed! Enjoy your journey! 🎉")

            except Exception as e:
                st.error("❌ An error occurred while generating the travel plan:")
                st.exception(e)
                st.info("Please check your `.env.local` file for correct API keys and model configurations. Also, check the terminal for detailed error messages from LiteLLM and CrewAI.")

if __name__ == "__main__":
    main()
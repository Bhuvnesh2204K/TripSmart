import os
import re
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

# Load environment variables using absolute path
# Ensure .env.local is in the same directory as this script, or adjust the path
env_path = os.path.join(os.getcwd(), ".env.local")
load_dotenv(dotenv_path=env_path)

# --- START OF ADDED DEBUG PRINTS (from Step 2) ---
print(f"\n--- Environment Variable Check ---")
hf_api_key_check = os.getenv('HUGGINGFACE_API_KEY')
print(f"DEBUG: HUGGINGFACE_API_KEY = {'*' * (len(hf_api_key_check) - 5) + hf_api_key_check[-5:] if hf_api_key_check else 'None (HF key not found)'}")
print(f"DEBUG: LITELLM_MODEL = {os.getenv('LITELLM_MODEL')}")
print(f"DEBUG: LITELLM_PROVIDER = {os.getenv('LITELLM_PROVIDER')}")
openai_api_key_check = os.getenv('OPENAI_API_KEY')
print(f"DEBUG: OPENAI_API_KEY = {'*' * (len(openai_api_key_check) - 5) + openai_api_key_check[-5:] if openai_api_key_check else 'None (OpenAI key not found)'}")
groq_api_key_check = os.getenv('GROQ_API_KEY')
print(f"DEBUG: GROQ_API_KEY = {'*' * (len(groq_api_key_check) - 5) + groq_api_key_check[-5:] if groq_api_key_check else 'None (Groq key not found)'}")
print(f"----------------------------------\n")
# --- END OF ADDED DEBUG PRINTS ---

# Set up Hugging Face API key
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not hf_api_key:
    print("WARNING: HUGGINGFACE_API_KEY not found in environment variables or is empty.")
    print("If you intend to use Hugging Face models, please set it in .env.local.")

# Clean invalid surrogate characters (for Windows terminal issues)
def clean_surrogates(text):
    if isinstance(text, str):
        return re.sub(r'[\ud800-\udfff]', '', text)
    return text

class TripAgents:
    def __init__(self):
        # Initialize the LLM once for all agents
        self.llm = self._initialize_llm()
        if not self.llm:
            # This check will now be hit if _initialize_llm raises an exception,
            # but it also handles the case if it somehow still returns None.
            raise RuntimeError("Failed to initialize any LLM. Please check your API keys and model configurations.")

    def _initialize_llm(self):
        """Initialize LLM with proper configuration, trying multiple providers/models."""

        # Set up environment variables for LiteLLM
        os.environ["LITELLM_LOG"] = os.getenv("LITELLM_LOG", "DEBUG")

        # Get model configuration from environment
        model_name = os.getenv("LITELLM_MODEL", "microsoft/DialoGPT-medium")
        provider = os.getenv("LITELLM_PROVIDER", "huggingface")

        # --- Attempt 1: Groq (Primary) ---
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                print("🤖 Initializing LLM with primary Groq model...")
                # Set environment variables for Groq
                os.environ['GROQ_API_KEY'] = groq_key
                llm = LLM(
                    model="llama3-8b-8192",
                    temperature=0.7,
                    max_tokens=1000,
                    custom_llm_provider="groq"
                )
                print("✅ Successfully initialized Groq model.")
                return llm
            except Exception as e3:
                print(f"❌ Groq failed: {e3}")
                print("Trying alternative Groq models...")
                
                # Try alternative Groq models
                groq_models = [
                    "mixtral-8x7b-32768",
                    "llama2-70b-4096",
                    "gemma-7b-it"
                ]
                for groq_model in groq_models:
                    try:
                        print(f"🔄 Trying alternative Groq model: {groq_model}")
                        llm = LLM(
                            model=groq_model,
                            temperature=0.7,
                            max_tokens=1000,
                            custom_llm_provider="groq"
                        )
                        print(f"✅ Successfully initialized alternative Groq model: {groq_model}")
                        return llm
                    except Exception as e4:
                        print(f"❌ Failed to initialize {groq_model}: {e4}")
                        continue
        else:
            print("Skipping Groq: GROQ_API_KEY not provided.")

        # --- Attempt 2: Hugging Face (Fallback) ---
        if hf_api_key:
            # IMPORTANT CHANGE: Pass only the model_name here.
            # LiteLLM will implicitly use the provider specified by LITELLM_PROVIDER env var.
            litellm_model_for_hf = model_name
            print(f"🔄 Trying Hugging Face as fallback (via {provider} provider): {litellm_model_for_hf}")
            try:
                llm = LLM(
                    model=litellm_model_for_hf, # Pass just the model name
                    api_key=hf_api_key,          # Explicitly pass the HF API key
                    temperature=0.7,
                    max_tokens=1000
                )
                print("✅ Successfully initialized Hugging Face model.")
                return llm
            except Exception as e:
                print(f"❌ Failed to initialize Hugging Face model {litellm_model_for_hf}: {e}")
                print("Trying alternative Hugging Face models...")

            # Attempt 2.1: Alternative Hugging Face models
            alternative_hf_models = [
                "google/flan-t5-base",
                "microsoft/DialoGPT-small",
                "HuggingFaceH4/zephyr-7b-beta"
            ]
            for alt_model in alternative_hf_models:
                try:
                    print(f"🔄 Trying alternative Hugging Face model: {alt_model}")
                    llm = LLM(
                        model=alt_model,          # Pass just the model name
                        api_key=hf_api_key,       # Always pass the HF key
                        temperature=0.7,
                        max_tokens=800
                    )
                    print(f"✅ Successfully initialized alternative Hugging Face model: {alt_model}")
                    return llm
                except Exception as e2:
                    print(f"❌ Failed to initialize {alt_model}: {e2}")
                    continue
        else:
            print("Skipping Hugging Face models: HUGGINGFACE_API_KEY not provided.")

        # --- Attempt 3: OpenAI (Fallback) ---
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                print("🔄 Trying OpenAI as fallback...")
                llm = LLM(
                    model="gpt-3.5-turbo", # Example OpenAI model (no explicit prefix needed, LiteLLM defaults to OpenAI)
                    api_key=openai_key,
                    temperature=0.7,
                    max_tokens=1000
                )
                print("✅ Successfully initialized OpenAI model.")
                return llm
            except Exception as e4:
                print(f"❌ OpenAI fallback failed: {e4}")
        else:
            print("Skipping OpenAI fallback: OPENAI_API_KEY not provided.")

        print("⚠️ All LLM initialization attempts failed. No LLM could be loaded.")
        # IMPORTANT CHANGE: Raise an error here if no LLM could be initialized
        raise RuntimeError("No valid LLM could be initialized. Please check your API keys and model configurations in .env.local.")

    def base_agent(self, role, goal, backstory):
        """Create base agent with proper LLM configuration"""
        agent_config = {
            'role': clean_surrogates(role),
            'goal': clean_surrogates(goal),
            'backstory': clean_surrogates(backstory),
            'verbose': True,
            'allow_delegation': False,  # Prevent agents from delegating to each other
            'max_iter': 5, # Increased iterations for more robust planning
        }

        # Add LLM if available
        if self.llm:
            agent_config['llm'] = self.llm
        else:
            # This print will technically be redundant if _initialize_llm raises an error,
            # but it's good for robustness if the condition is ever met otherwise.
            print("WARNING: LLM not initialized for agents. They might use default or fail.")

        return Agent(**agent_config)

    def city_selector_agent(self):
        return self.base_agent(
            role='City Selection Expert',
            goal='Identify the 3 best cities to visit based on user preferences including travel type, interests, and season',
            backstory="""You are an expert travel geographer with extensive knowledge of global destinations.
            You specialize in matching travelers with perfect cities based on their preferences, interests, and travel seasons.
            You always provide exactly 3 city recommendations with clear reasoning, covering various aspects like culture, adventure, and relaxation."""
        )

    def local_expert_agent(self):
        return self.base_agent(
            role='Local Destination Expert',
            goal='Provide comprehensive insights about the selected city including attractions, culture, and practical tips',
            backstory="""You are a knowledgeable local guide with insider knowledge about cities worldwide.
            You provide detailed information about attractions, local customs, cuisine, and practical travel tips
            that help visitors have an authentic and enjoyable experience."""
        )

    def travel_planner_agent(self):
        return self.base_agent(
            role='Professional Travel Planner',
            goal='Create detailed, practical day-by-day itineraries that maximize the travel experience',
            backstory="""You are an experienced professional travel planner who creates efficient,
            well-organized itineraries. You consider travel time, opening hours, proximity of attractions,
            and optimal scheduling to create the best possible travel experience."""
        )

    def budget_manager_agent(self):
        return self.base_agent(
            role='Travel Budget Specialist',
            goal='Create realistic budget plans that balance cost-effectiveness with quality experiences',
            backstory="""You are a financial planning expert specializing in travel budgets.
            You help travelers get the most value for their money while ensuring they don't overspend.
            You consider all aspects of travel costs and provide practical budget breakdowns."""
        )

class TripTasks:
    def city_selection_task(self, agent, inputs):
        description = f"""
        Based on the following user preferences, select the 3 best cities to visit:

        Travel Type: {inputs.get('travel_type', 'Not specified')}
        Interests: {inputs.get('interests', 'Not specified')}
        Season: {inputs.get('season', 'Not specified')}
        Budget Range: {inputs.get('budget', 'Not specified')}
        Duration: {inputs.get('duration', 'Not specified')} days

        For each city, provide:
        1. City name and country
        2. Why it matches their travel type and interests
        3. Why it's perfect for the specified season
        4. Brief highlight of what makes it special

        Format your response as a numbered list with city names clearly stated.
        """

        return Task(
            description=clean_surrogates(description),
            agent=agent,
            expected_output=clean_surrogates(
                "A numbered list of exactly 3 cities with clear city names and 2-3 sentence explanations for each, "
                "explaining why each city is perfect for the user's preferences. Example: '1. Paris, France - ...'"
            )
        )

    def city_research_task(self, agent, city):
        description = f"""
        Provide comprehensive research about {city} including:

        1. TOP 5 ATTRACTIONS: Must-visit places with brief descriptions and estimated time to visit each.
        2. LOCAL CUISINE: Signature dishes and where to find them (e.g., specific types of restaurants or markets).
        3. CULTURAL INSIGHTS: Important customs, etiquette, and cultural norms (e.g., tipping, greetings).
        4. ACCOMMODATION: Best areas to stay for different budgets (e.g., luxury, mid-range, budget) and types of lodging.
        5. TRANSPORTATION: How to get around the city efficiently (public transport, taxis, ride-shares) and airport transfers.
        6. BEST TIME TO VISIT: Optimal months, what to expect regarding weather and crowds, and major events if any.
        7. PRACTICAL TIPS: Essential information for first-time visitors (e.g., currency, safety, language basics, common scams).

        Make this practical and actionable for travelers, using bullet points or numbered lists where appropriate for clarity.
        """

        return Task(
            description=clean_surrogates(description),
            agent=agent,
            expected_output=clean_surrogates(
                "Well-organized sections with clear headings (e.g., 'TOP 5 ATTRACTIONS', 'LOCAL CUISINE'), "
                "using bullet points or numbered lists, and practical, actionable information "
                "that helps travelers understand and navigate the destination effectively. "
                "Include estimated costs or price ranges where applicable for a specific budget range if mentioned in inputs."
            )
        )

    def itinerary_creation_task(self, agent, inputs, city):
        duration = inputs.get('duration', '3')
        travel_type = inputs.get('travel_type', 'leisure')
        interests = inputs.get('interests', 'general sightseeing')

        description = f"""
        Create a detailed {duration}-day itinerary for {city} based on:
        - Travel Type: {travel_type}
        - Interests: {interests}
        - Duration: {duration} days
        - Current location: Surat, Gujarat, India (Consider this for travel logistics if applicable, but focus on the destination city itself)

        For each day, include:
        1. DAY NUMBER: Clearly state the day (e.g., 'Day 1: Arrival and City Exploration')
        2. Morning activities (9 AM - 12 PM) with specific attractions/locations
        3. Afternoon activities (12 PM - 5 PM) with specific attractions/locations
        4. Evening activities (5 PM - 9 PM) including recommendations for dinner/nightlife
        5. Recommended restaurants for each meal (Breakfast, Lunch, Dinner) with type of cuisine
        6. Estimated transportation methods between locations (e.g., 'walk', 'metro', 'taxi')
        7. Estimated time for each activity
        8. Any booking requirements or tips (e.g., 'Book in advance', 'Wear comfortable shoes')

        Make sure activities are geographically logical, account for realistic travel time between locations,
        and maximize the experience given the specified travel type and interests.
        """

        return Task(
            description=clean_surrogates(description),
            agent=agent,
            expected_output=clean_surrogates(
                f"A comprehensive day-by-day itinerary for {duration} days, clearly structured by day and time slots. "
                "Each entry should include specific activities, recommended restaurants, transportation methods, "
                "estimated timings, and practical tips, presented in a highly readable and actionable format."
            )
        )

    def budget_planning_task(self, agent, inputs, city):
        budget_range = inputs.get('budget', 'moderate')
        duration = inputs.get('duration', '3')
        travel_type = inputs.get('travel_type', 'leisure')

        description = f"""
        Create a realistic budget plan for a {duration}-day trip to {city} assuming a {budget_range} budget.
        Consider that the traveler is from Surat, Gujarat, India, which might influence spending habits or expectations.

        Break down costs for the following categories:
        1. ACCOMMODATION: Per night costs and total for {duration} nights. Provide options for the given budget.
        2. MEALS: Daily averages for Breakfast, Lunch, Dinner. Provide typical costs for different meal types.
        3. TRANSPORTATION: Local transport (daily passes, single rides), airport transfers (round trip), and potentially inter-city travel if applicable.
        4. ATTRACTIONS: Entry fees for major sights and activities.
        5. SHOPPING & MISCELLANEOUS: Estimated daily allowance for souvenirs, snacks, coffee, and small purchases.
        6. EMERGENCY FUND: A recommended 10-15% buffer of the total estimated cost.

        Provide:
        - Daily budget estimates for each category.
        - Total estimated budget for the entire {duration}-day trip.
        - Money-saving tips specific to {city}.
        - Splurge recommendations (e.g., a fancy dinner, unique experience) with estimated costs.

        Use currency relevant to the destination city or provide a clear indication (e.g., USD, EUR, local currency).
        """

        return Task(
            description=clean_surrogates(description),
            agent=agent,
            expected_output=clean_surrogates(
                "An itemized budget breakdown presented in a clear table or list format, "
                "with daily and total estimated costs for each category (accommodation, meals, transport, etc.). "
                "Include currency, money-saving tips, and optional splurge recommendations with price ranges."
            )
        )


class TripCrew:
    def __init__(self, inputs):
        self.inputs = inputs
        self.agents = TripAgents() # This will now raise an error if LLM init fails
        self.tasks = TripTasks()

    def extract_first_city(self, city_output):
        """Extract the first city from the city selection output"""
        if isinstance(city_output, dict):
            city_output = str(city_output)

        # Clean the output first
        city_output = clean_surrogates(city_output)

        # Try different patterns to extract city names
        patterns = [
            r"1\.\s*([A-Z][a-zA-Z\s]+?)(?:,\s*[A-Z][a-zA-Z\s]+?|\s*[-,:]|\n)",  # "1. Paris, France" or "1. New York"
            r"[-*•]\s*([A-Z][a-zA-Z\s]+?)(?:,\s*[A-Z][a-zA-Z\s]+?|\s*[-,:]|\n)",  # Bullet points
            r"(\b[A-Z][a-zA-Z\s]+?)(?:\s*[-,:])",  # City names followed by punctuation
            r"([A-Z][a-zA-Z\s]{3,25})"  # Any capitalized words (cities), minimum 3 letters
        ]

        for pattern in patterns:
            matches = re.findall(pattern, city_output)
            if matches:
                for match in matches:
                    city = match.strip()
                    # Filter out common non-city words or partial matches
                    excluded_words = ['the', 'and', 'for', 'with', 'city', 'travel', 'trip', 'based', 'type', 'perfect', 'best', 'experience', 'plan', 'planning', 'expert', 'reasons', 'selection']
                    if (len(city) > 2 and
                        city.lower() not in excluded_words and
                        not any(city.lower().startswith(ew) for ew in excluded_words) and
                        not city.lower().endswith('city')): # Exclude generic "city"
                        print(f"DEBUG: Extracted potential city: {city}")
                        return city

        # Default fallback
        print("⚠️ Could not extract a valid city from output, using default: Paris")
        return "Paris"

    def run(self):
        try:
            print("🌍 Starting city selection...")

            # Step 1: City Selection
            selector = self.agents.city_selector_agent()
            city_task = self.tasks.city_selection_task(selector, self.inputs)

            city_crew = Crew(
                agents=[selector],
                tasks=[city_task],
                verbose=True,
                full_output=True # Ensures we get a detailed output object
            )

            city_result_object = city_crew.kickoff()

            # Ensure we get the raw string content for extraction
            city_output_raw = city_result_object.raw if hasattr(city_result_object, 'raw') else str(city_result_object.result)
            print(f"City selection raw output: {city_output_raw}")

            # Extract selected city
            selected_city = self.extract_first_city(city_output_raw)
            print(f"✅ Selected city for detailed planning: {selected_city}")

            # Step 2: Research, Itinerary, and Budget Planning
            print(f"🔍 Researching {selected_city}...")

            researcher = self.agents.local_expert_agent()
            planner = self.agents.travel_planner_agent()
            budgeter = self.agents.budget_manager_agent()

            research_task = self.tasks.city_research_task(researcher, selected_city)
            itinerary_task = self.tasks.itinerary_creation_task(planner, self.inputs, selected_city)
            budget_task = self.tasks.budget_planning_task(budgeter, self.inputs, selected_city)

            trip_crew = Crew(
                agents=[researcher, planner, budgeter],
                tasks=[research_task, itinerary_task, budget_task],
                verbose=True,
                full_output=True # Again, ensure full output
            )

            detailed_results_object = trip_crew.kickoff()

            # Extract results properly from tasks_output
            city_research_output = "No research available"
            itinerary_output = "No itinerary available"
            budget_output = "No budget available"

            if hasattr(detailed_results_object, 'tasks_output') and detailed_results_object.tasks_output:
                for task_result in detailed_results_object.tasks_output:
                    if "research" in task_result.description.lower(): # Crude check for research task
                        city_research_output = task_result.raw if hasattr(task_result, 'raw') else str(task_result)
                    elif "itinerary" in task_result.description.lower(): # Crude check for itinerary task
                        itinerary_output = task_result.raw if hasattr(task_result, 'raw') else str(task_result)
                    elif "budget" in task_result.description.lower(): # Crude check for budget task
                        budget_output = task_result.raw if hasattr(task_result, 'raw') else str(task_result)
            else:
                 # Fallback if tasks_output is not structured as expected
                print("WARNING: 'tasks_output' not found as expected. Attempting string parsing fallback.")
                result_str = str(detailed_results_object.result if hasattr(detailed_results_object, 'result') else detailed_results_object)
                # This parsing is less reliable but better than nothing
                city_research_output = result_str
                itinerary_output = result_str
                budget_output = result_str


            return {
                "🌆 City Selection": clean_surrogates(city_output_raw),
                "🗺 City Research": clean_surrogates(city_research_output),
                "📅 Itinerary": clean_surrogates(itinerary_output),
                "💸 Budget Plan": clean_surrogates(budget_output)
            }

        except Exception as e:
            print(f"❌ Error in TripCrew.run(): {e}")
            import traceback
            traceback.print_exc()

            return {
                "🌆 City Selection": f"Error occurred during city selection: {str(e)}. Please check your API configuration and try again.",
                "🗺 City Research": "Unable to generate due to error",
                "📅 Itinerary": "Unable to generate due to error",
                "💸 Budget Plan": "Unable to generate due to error"
            }
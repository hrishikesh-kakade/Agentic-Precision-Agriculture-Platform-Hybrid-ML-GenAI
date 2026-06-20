import os
from groq import Groq
from config import GROQ_API_KEY

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "openai/gpt-oss-20b" 

def run_soil_agent(N, P, K, ph):
    prompt = f"""
    You are an expert Soil Chemist. Analyze these metrics:
    Nitrogen (N): {N}, Phosphorus (P): {P}, Potassium (K): {K}, pH: {ph}.
    Identify specific nutrient deficiencies or toxicities. Provide concise chemical recommendations to stabilize the soil.
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_NAME,
        temperature=0.2
    )
    return response.choices[0].message.content

def run_pathology_agent(disease_prediction):
    prompt = f"""
    You are a Plant Pathologist. The vision model has diagnosed: '{disease_prediction}'.
    List treatment guidelines for this condition. Highlight any biochemical components or actions that could worsen this outbreak.
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_NAME,
        temperature=0.2
    )
    return response.choices[0].message.content

def run_weather_agent(temp, humidity, rainfall):
    prompt = f"""
    You are an Agricultural Meteorologist. Conditions: Temp: {temp}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm.
    Evaluate how these conditions impact operations (e.g., risk of fertilizer leaching, optimal spray timing).
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_NAME,
        temperature=0.2
    )
    return response.choices[0].message.content

def run_coordinator_jury(soil_report, pathology_report, weather_report):
    jury_prompt = f"""
    You are the Chief Agricultural Coordinator. Reconcile these reports into a final prescription.
    You MUST format your output exactly using the following section tags so our system can parse it:

    [SOIL]
    (Consolidated soil nutrient blueprint)

    [PATHOLOGY]
    (Disease management and biosecurity warnings)

    [WEATHER]
    (Localized meteorological timing advice)

    [PRESCRIPTION]
    (Final, safe, step-by-step action plan)

    Reports to evaluate:
    {soil_report}
    {pathology_report}
    {weather_report}
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": jury_prompt}],
        model=MODEL_NAME,
        temperature=0.4 
    )
    return response.choices[0].message.content

def generate_agentic_prescription(N, P, K, ph, temp, humidity, rainfall, disease_prediction):
    soil_analysis = run_soil_agent(N, P, K, ph)
    pathology_analysis = run_pathology_agent(disease_prediction)
    weather_analysis = run_weather_agent(temp, humidity, rainfall)
    final_prescription = run_coordinator_jury(soil_analysis, pathology_analysis, weather_analysis)
    
    return final_prescription
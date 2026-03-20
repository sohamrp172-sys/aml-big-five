"""Streamlit UI for Big Five Personality Analyzer."""

import streamlit as st
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

# IPIP-50 short question labels per item
QUESTIONS = {
    "EXT1": "I am the life of the party",
    "EXT2": "I don't talk a lot",
    "EXT3": "I feel comfortable around people",
    "EXT4": "I keep in the background",
    "EXT5": "I start conversations",
    "EXT6": "I have little to say",
    "EXT7": "I talk to a lot of different people at parties",
    "EXT8": "I don't like to draw attention to myself",
    "EXT9": "I don't mind being the center of attention",
    "EXT10": "I am quiet around strangers",
    "EST1": "I get stressed out easily",
    "EST2": "I am relaxed most of the time",
    "EST3": "I worry about things",
    "EST4": "I seldom feel blue",
    "EST5": "I am easily disturbed",
    "EST6": "I get upset easily",
    "EST7": "I change my mood a lot",
    "EST8": "I have frequent mood swings",
    "EST9": "I get irritated easily",
    "EST10": "I often feel blue",
    "AGR1": "I feel little concern for others",
    "AGR2": "I am interested in people",
    "AGR3": "I insult people",
    "AGR4": "I sympathize with others' feelings",
    "AGR5": "I am not interested in others' problems",
    "AGR6": "I have a soft heart",
    "AGR7": "I am not really interested in others",
    "AGR8": "I take time out for others",
    "AGR9": "I feel others' emotions",
    "AGR10": "I make people feel at ease",
    "CSN1": "I am always prepared",
    "CSN2": "I leave my belongings around",
    "CSN3": "I pay attention to details",
    "CSN4": "I make a mess of things",
    "CSN5": "I get chores done right away",
    "CSN6": "I often forget to put things back",
    "CSN7": "I like order",
    "CSN8": "I shirk my duties",
    "CSN9": "I follow a schedule",
    "CSN10": "I am exacting in my work",
    "OPN1": "I have a rich vocabulary",
    "OPN2": "I have difficulty understanding abstract ideas",
    "OPN3": "I have a vivid imagination",
    "OPN4": "I am not interested in abstract ideas",
    "OPN5": "I have excellent ideas",
    "OPN6": "I do not have a good imagination",
    "OPN7": "I am quick to understand things",
    "OPN8": "I use difficult words",
    "OPN9": "I spend time reflecting on things",
    "OPN10": "I am full of ideas",
}

TRAITS = {
    "EXT": "Extraversion",
    "EST": "Neuroticism",
    "AGR": "Agreeableness",
    "CSN": "Conscientiousness",
    "OPN": "Openness",
}

TRAIT_DESCRIPTIONS = {
    "Extraversion": "Reflects sociability, assertiveness, and positive emotions. High scorers are outgoing and energetic; low scorers are reserved and independent.",
    "Neuroticism": "Reflects emotional instability and tendency toward negative emotions. High scorers experience more stress and mood swings; low scorers are calm and resilient.",
    "Agreeableness": "Reflects cooperativeness, trust, and empathy. High scorers are compassionate and helpful; low scorers are more competitive and skeptical.",
    "Conscientiousness": "Reflects organization, dependability, and self-discipline. High scorers are methodical and reliable; low scorers are flexible and spontaneous.",
    "Openness": "Reflects curiosity, creativity, and openness to new experiences. High scorers are imaginative and adventurous; low scorers prefer routine and practicality.",
}

SCORE_INTERPRETATIONS = {
    "Extraversion": {
        "high": "Outgoing, energetic, sociable",
        "mid": "Balanced between social and solitary",
        "low": "Reserved, independent, reflective",
    },
    "Neuroticism": {
        "high": "Emotionally reactive, stress-prone",
        "mid": "Moderate emotional sensitivity",
        "low": "Calm, emotionally stable, resilient",
    },
    "Agreeableness": {
        "high": "Compassionate, cooperative, trusting",
        "mid": "Balanced between empathy and assertiveness",
        "low": "Competitive, skeptical, direct",
    },
    "Conscientiousness": {
        "high": "Organized, disciplined, reliable",
        "mid": "Moderately organized and flexible",
        "low": "Spontaneous, flexible, carefree",
    },
    "Openness": {
        "high": "Creative, curious, adventurous",
        "mid": "Open yet grounded in practicality",
        "low": "Practical, conventional, routine-oriented",
    },
}


def interpret_score(trait: str, score: float) -> str:
    interp = SCORE_INTERPRETATIONS[trait]
    if score > 0.5:
        return interp["high"]
    elif score < -0.5:
        return interp["low"]
    return interp["mid"]


def draw_radar_chart(scores: dict):
    labels = list(scores.keys())
    values = list(scores.values())

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="steelblue", alpha=0.25)
    ax.plot(angles, values, color="steelblue", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_title("Big Five Personality Profile", size=14, pad=20)
    ax.grid(True)
    return fig


def main():
    st.set_page_config(page_title="Big Five Personality Analyzer", layout="wide")
    st.title("Big Five Personality Analyzer")
    st.write("Rate each statement from **1** (Strongly Disagree) to **5** (Strongly Agree).")

    responses = {}

    for prefix, trait_name in TRAITS.items():
        st.subheader(f"{trait_name}")
        cols = st.columns(2)
        for i in range(1, 11):
            key = f"{prefix}{i}"
            col = cols[0] if i <= 5 else cols[1]
            with col:
                responses[key] = st.slider(
                    f"{key}: {QUESTIONS[key]}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=key,
                )

    st.divider()

    if st.button("Analyze My Personality", type="primary", use_container_width=True):
        try:
            resp = requests.post(f"{API_URL}/predict", json=responses, timeout=10)
            resp.raise_for_status()
            scores = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
            return
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")
            return

        st.subheader("Your Results")

        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            fig = draw_radar_chart(scores)
            st.pyplot(fig)
            plt.close(fig)

        with col_table:
            st.markdown("#### Score Breakdown")
            for trait, score in scores.items():
                interp = interpret_score(trait, score)
                direction = "+" if score >= 0 else ""
                st.markdown(f"**{trait}**: `{direction}{score:.2f}` — {interp}")

        with st.expander("What does this mean?"):
            for trait, desc in TRAIT_DESCRIPTIONS.items():
                st.markdown(f"**{trait}**: {desc}")


if __name__ == "__main__":
    main()

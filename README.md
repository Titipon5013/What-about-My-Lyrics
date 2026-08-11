# What about My Lyrics (WAML)
**AI-Powered Lyrics Analyzer & Hit Potential Engine**

What about My Lyrics (WAML) is a Streamlit-based web application designed to help songwriters, artists, and beat makers analyze Thai and English lyrics. It utilizes Natural Language Processing (NLP) to extract emotional vibes, evaluate song arrangements, and provide data-driven beat recommendations such as BPM, Key, and Scale to maximize a track's commercial hit potential.

## Key Features

- **Deep Lyrics Sentiment Analysis:** Detects the emotional vibe (Melancholy, Upbeat, Aggressive, Seductive) using a custom rule-based Thai/English lexicon and NLP tokenization.
- **AI Songwriting Insights:** Scans for earworm potential (repetition), bilingual flow (code-mixing), and thematic storytelling (metaphors) to provide actionable feedback.
- **Interactive Song Loop Selector:** Allows users to arrange song sections, such as Intro, Verse, and Hook, like building blocks to evaluate structural dynamics.
- **Hit Potential Score:** Calculates a heuristic percentage based on the strength of the lyrical vibe, songwriting techniques, and the commercial viability of the song's arrangement.
- **Data-Driven Producer Briefing:** Suggests the optimal BPM, Target Key, and Musical Mode by cross-referencing the detected vibe and genre with real-world music statistics.

## Tech Stack

- **Frontend & UI:** Streamlit
- **Data Manipulation:** Pandas
- **Natural Language Processing (Thai):** PyThaiNLP

## Installation & Local Setup

1. Clone the repository:

	```bash
	git clone https://github.com/Titipon5013/What-about-My-Lyrics.git
	cd What-about-My-Lyrics
	```

2. Install the required dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Ensure the dataset is in the correct directory:

	```
	data_resources/high_popularity_spotify_data.csv
	```

4. Run the Streamlit application:

	```bash
	streamlit run app.py
	```

## Dataset Acknowledgement

The statistical mappings for BPM, Key, and Musical Modes used in this project are derived from the **Spotify Music Dataset**.
Special thanks to **Solomon Ameh** for providing this comprehensive dataset.

- **Dataset Link:** [Spotify Music Dataset on Kaggle](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset)

## Author

Developed by **Titipon Tawong**
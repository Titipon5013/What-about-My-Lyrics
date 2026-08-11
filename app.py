import streamlit as st
import pandas as pd
import re
from collections import Counter
from pythainlp.tokenize import word_tokenize

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(
    page_title="What about My Lyrics",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:ital,wght@0,400;0,700;1,400&family=Permanent+Marker&display=swap');

    .main-title {
        font-family: 'Permanent Marker', cursive;
        text-align: center;
        font-size: 4rem;
        color: #ff3366;
        text-shadow: 3px 3px 0px #000, -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
        margin-bottom: -10px;
        letter-spacing: 2px;
    }

    .sub-title {
        font-family: 'Kanit', sans-serif;
        text-align: center;
        font-size: 1.2rem;
        color: #fca311;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 30px;
    }

    div[data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 2px solid #ff3366;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 4px 4px 0px #ff3366;
    }

    .flow-box {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.2rem;
        color: #fca311;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        border: 1px dashed #ff3366;
    }

    .potential-score {
        font-size: 4rem;
        font-weight: bold;
        color: #00ffcc;
        text-align: center;
        text-shadow: 2px 2px 10px rgba(0,255,204,0.5);
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. Data & Deep NLP Engine Setup
# ==========================================
@st.cache_data
def load_spotify_data():
    try:
        return pd.read_csv('data_resources/high_popularity_spotify_data.csv')
    except Exception:
        return pd.DataFrame(columns=['playlist_genre', 'mode', 'tempo', 'key'])


df_high = load_spotify_data()

lexicon_db = {
    "Melancholy 🌧️": ['เจ็บ', 'ร้องไห้', 'ลืม', 'ทิ้ง', 'เศร้า', 'น้ำตา', 'ทรมาน', 'อ้อนวอน', 'แตกสลาย', 'อ้างว้าง',
                      'sad', 'cry', 'broken', 'hurt', 'alone', 'tears', 'pain', 'fade', 'lonely'],
    "Upbeat / Party 🪩": ['สนุก', 'รัก', 'ยิ้ม', 'ปาร์ตี้', 'เต้น', 'สดใส', 'ชนแก้ว', 'เมา', 'สุดเหวี่ยง', 'คืนนี้',
                         'ฉลอง', 'happy', 'love', 'smile', 'party', 'dance', 'tonight', 'vibes', 'cheers', 'high'],
    "Aggressive / Flex 🔥": ['รวย', 'เดือด', 'ของจริง', 'รันวงการ', 'แชมป์', 'สู้', 'ศัตรู', 'ทอง', 'เงิน', 'อำนาจ',
                            'flex', 'money', 'hater', 'gang', 'boss', 'rich', 'grind', 'hustle', 'hood', 'crew',
                            'opps'],
    "Seductive / Romance 🌹": ['เสน่ห์', 'หลง', 'จูบ', 'สัมผัส', 'ร้อนแรง', 'คืนนี้', 'ต้องการ', 'กลิ่นหอม', 'สบตา',
                              'เซ็กซี่', 'ไฟ', 'อันตราย', 'sexy', 'shawty', 'my lady', 'baby', 'kiss', 'touch',
                              'desire', 'body', 'babe', 'toxic']
}


def clean_and_tokenize(text):
    text = text.lower()
    return word_tokenize(text, engine='newmm', keep_whitespace=False)


def analyze_vibe_score(lyrics_text):
    tokens = clean_and_tokenize(lyrics_text)
    scores = {category: 0 for category in lexicon_db.keys()}
    for word in tokens:
        for category, keywords in lexicon_db.items():
            if word in keywords:
                scores[category] += 1
    best_vibe = max(scores, key=scores.get) if any(scores.values()) else "Neutral ☁️"
    return best_vibe, scores


# --- วิเคราะห์เทคนิคการแต่งเพลงเชิงลึก (Songwriting Analysis) ---
def analyze_lyric_techniques(lyrics_text, detected_vibe="Neutral ☁️"):
    insights = []

    # 1. Earworm / Repetition & Storytelling Check
    lines = [line.strip() for line in lyrics_text.split('\n') if len(line.strip()) > 5]
    line_counts = Counter(lines)
    repeats = [line for line, count in line_counts.items() if count > 1]

    if repeats:
        insights.append(
            f"🪝 **Earworm Potential:** มีการใช้ประโยคซ้ำเช่น *'{repeats[0]}'* ท่อนนี้มีศักยภาพสูงมากที่จะเป็น 'ท่อนจำ' แนะนำให้วางไว้ใน Hook")
    else:
        # ถ้าเพลงยาว (เล่าเรื่อง) และมี Vibe หม่นหมอง ไม่ต้องบังคับให้ใส่ประโยคซ้ำ
        if len(lines) > 8 and "Melancholy" in detected_vibe:
            insights.append(
                "📝 **Storytelling Flow:** เนื้อเพลงมีการเล่าเรื่องที่ลึกซึ้งและส่งผ่านความรู้สึกได้ดี เพลงสไตล์นี้ดึงคนฟังให้อินไปกับบรรยากาศได้โดยไม่ต้องพึ่งประโยคจำซ้ำๆ!")
        else:
            insights.append(
                "💡 **Catchy Tip:** ลองหาคีย์เวิร์ดเด็ดๆ มาร้องซ้ำย้ำๆ ในท่อนฮุคดูครับ จะช่วยให้เพลงติดหูและกลายเป็นไวรัลได้ง่ายขึ้น")

    # 2. Code-Mixing (Bilingual Flow)
    eng_words = re.findall(r'[a-zA-Z]+', lyrics_text.lower())
    # กรองคำที่เป็น Header ของโครงสร้างเพลงออก
    exclude_words = {'hook', 'verse', 'rap', 'intro', 'outro', 'bridge', 'pre', 'solo'}
    filtered_eng = [w for w in eng_words if w not in exclude_words]

    if len(filtered_eng) > 5:
        unique_eng = list(set(filtered_eng))[:4]
        insights.append(
            f"🌐 **Bilingual Flow:** ดีมาก! มีการสลับภาษา (Code-mixing) เช่น *{', '.join(unique_eng)}* ช่วยให้ Flow การร้องดูลื่นไหล อินเตอร์ และเข้ากับบีทยุคใหม่")

    # 3. Thematic Metaphors (จัดหมวดหมู่กลุ่มคำเปรียบเปรยให้หลากหลาย)
    metaphor_themes = {
        "Space & Sci-Fi 🚀": ['อวกาศ', 'ดาว', 'ดวงจันทร์', 'พระอาทิตย์', 'จักรวาล', 'galaxy', 'light', 'moon', 'star',
                             'โลก', 'พลูโต', 'pluto'],
        "Nature & Elements 🌪️": ['ฝน', 'พายุ', 'ลม', 'ไฟ', 'ทะเล', 'ภูเขา', 'น้ำตาฟ้า', 'ฤดู', 'rain', 'fire', 'storm',
                                 'ocean', 'คลื่น'],
        "Dark & Pain 🥀": ['พิษ', 'บาดแผล', 'มีด', 'เลือด', 'แตกสลาย', 'นรก', 'toxic', 'pain', 'scar', 'ปีศาจ', 'จมน้ำ',
                          'ความตาย', 'ตาย', 'พังทลาย'],
        "Urban & Street 🏙️": ['ถนน', 'แสงสี', 'เมือง', 'ควัน', 'เงิน', 'ทอง', 'city', 'street', 'smoke', 'hood',
                              'แสงไฟ', 'รถ', 'ขับ'],
        "Seductive & Danger 🍷": ['อันตราย', 'ร้อนแรง', 'เผา', 'หลุมพราง', 'ยาพิษ', 'รสชาติ', 'danger', 'burning',
                                 'taste', 'secret', 'ความลับ']
    }

    found_themes = {}
    for theme, words in metaphor_themes.items():
        matched = [w for w in words if w.lower() in lyrics_text.lower()]
        if matched:
            found_themes[theme] = matched

    if found_themes:
        # หา Theme ที่เนื้อเพลงนี้ใช้งานเยอะที่สุด
        best_theme = max(found_themes, key=lambda k: len(found_themes[k]))
        matched_words = found_themes[best_theme][:4]  # ดึงมาโชว์แค่ 4 คำ
        insights.append(
            f"🎨 **Thematic Storytelling ({best_theme}):** คุมโทนเนื้อหาได้เฉียบคม! มีการใช้กลุ่มคำเปรียบเปรย เช่น *{', '.join(matched_words)}* สร้างภาพลักษณ์ (Visual) ในหัวคนฟังได้ชัดเจน ทำให้เพลงมีเสน่ห์น่าค้นหา")

    return insights

def evaluate_song_structure(flow_sequence):
    insights = []
    if not flow_sequence:
        return ["🚨 กรุณาจัดเรียง Loop เพลงอย่างน้อย 1 ท่อน"]
    if "Hook" not in flow_sequence:
        insights.append("🚨 **Warning:** เพลงนี้ไม่มีท่อน Hook! ฮุคคือหัวใจหลักที่ทำให้เพลงฮิต")
    if flow_sequence and flow_sequence[0] == "Hook":
        insights.append("⏩ **Structure Idea:** เปิดเพลงด้วย Hook เป็นเทคนิคยอดฮิต ดึงความสนใจคนฟังได้ใน 5 วินาทีแรก!")
    for i in range(len(flow_sequence) - 1):
        if flow_sequence[i] == "Pre-Hook" and flow_sequence[i + 1] == "Hook":
            insights.append(
                "📈 **Good Flow:** การวาง Pre-Hook ไต่ระดับอารมณ์เข้า Hook จะช่วยให้บีทระเบิดพลังได้สุดยอดมาก")
            break
    if not insights:
        insights.append("✅ โครงสร้างเพลง (Arrangement) ดูสมดุลและไหลลื่นดีครับ")
    return insights


def generate_producer_brief(genre, detected_vibe, df):
    target_music_mode = 0 if "Melancholy" in detected_vibe or "Aggressive" in detected_vibe else 1
    mask = (df['playlist_genre'] == genre) & (df['mode'] == target_music_mode)
    trend_data = df[mask]

    if trend_data.empty:
        trend_data = df[df['playlist_genre'] == genre]
    if trend_data.empty:
        return {"Target_Tempo": 115, "Target_Key": "C#", "Target_Mode": "Minor"}

    rec_tempo = round(trend_data['tempo'].median())
    rec_key_val = trend_data['key'].mode()[0] if not trend_data['key'].mode().empty else 1
    key_mapping = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#',
                   11: 'B'}

    return {
        "Target_Tempo": rec_tempo,
        "Target_Key": key_mapping.get(rec_key_val, str(rec_key_val)),
        "Target_Mode": "Minor" if target_music_mode == 0 else "Major"
    }


def calculate_popularity_potential(vibe, flow_sequence, lyrics_text):
    score = 40
    if vibe != "Neutral ☁️": score += 10
    if "Hook" in flow_sequence: score += 15
    if flow_sequence and flow_sequence[0] == "Hook": score += 10
    for i in range(len(flow_sequence) - 1):
        if flow_sequence[i] == "Pre-Hook" and flow_sequence[i + 1] == "Hook":
            score += 10
            break
    if 3 <= len(flow_sequence) <= 7: score += 5

    # บวกคะแนนเพิ่มถ้ามีเทคนิคการแต่งเพลงที่ดี
    eng_words = re.findall(r'[a-zA-Z]+', lyrics_text)
    if len(eng_words) > 0: score += 5
    lines = [line.strip() for line in lyrics_text.split('\n') if len(line.strip()) > 5]
    if len([l for l, c in Counter(lines).items() if c > 1]) > 0: score += 4

    return min(score, 99)


# ==========================================
# 3. Dynamic Session State
# ==========================================
if 'loop_sequence' not in st.session_state:
    st.session_state.loop_sequence = []


def add_to_sequence(section):
    st.session_state.loop_sequence.append(section)


def clear_sequence():
    st.session_state.loop_sequence = []


def pop_sequence():
    if st.session_state.loop_sequence:
        st.session_state.loop_sequence.pop()


# ==========================================
# 4. App UI Layout
# ==========================================
st.markdown("<h1 class='main-title'>WHAT ABOUT MY LYRICS</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>AI Lyrics Analyzer & Hit Potential Engine</p>", unsafe_allow_html=True)

st.markdown("### ⚙️ 1. Project Setup")
col_setup1, col_setup2 = st.columns(2)
with col_setup1:
    genre = st.selectbox("🎸 Main Genre", [
        "hip-hop", "r&b", "pop", "electronic", "rock", "indie", "afrobeats", "k-pop", "latin"
    ], index=1)
with col_setup2:
    subgenre = st.selectbox("🎛️ Subgenre", [
        "trap", "drill", "jersey club", "new jazz", "boom bap", "soul", "dancehall", "modern"
    ], index=0)

st.divider()

st.markdown("### 📝 2. Drop Your Lyrics")
full_lyrics = st.text_area("เนื้อเพลง", height=200, label_visibility="collapsed",
                           placeholder="วางเนื้อเพลงทั้งหมดของคุณที่นี่...\n(ระบบจะวิเคราะห์หาท่อนฮิต, สแลง, และการคุมโทน)")

st.divider()

st.markdown("### 🔁 3. Song Loop Selector")
col_opts1, col_opts2 = st.columns([3, 1])
with col_opts1:
    section_options = ["Intro", "Verse", "Pre-Hook", "Hook", "Rap", "Bridge", "Outro", "Solo"]
    selected_sec = st.selectbox("เลือกท่อนเพลง", section_options, label_visibility="collapsed")
with col_opts2:
    if st.button("➕ เพิ่มเข้า Loop", use_container_width=True):
        add_to_sequence(selected_sec)

ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
with ctrl1:
    if st.button("⬅️ ลบล่าสุด"): pop_sequence()
with ctrl2:
    if st.button("🗑️ ล้างทั้งหมด"): clear_sequence()

if st.session_state.loop_sequence:
    flow_str = " ➡️ ".join(st.session_state.loop_sequence)
    st.markdown(f"<div class='flow-box'>{flow_str}</div>", unsafe_allow_html=True)
else:
    st.info("💡 จัดเรียงโครงสร้างโดยการเพิ่มท่อนเพลง (เช่น Intro ➡️ Verse ➡️ Pre-Hook ➡️ Hook)")

st.divider()

# ==========================================
# 5. Execute Analysis
# ==========================================
if st.button("🔥 ANALYZE HIT POTENTIAL", type="primary", use_container_width=True):

    if not full_lyrics:
        st.error("Hold up! Drop some lyrics first. 🛑")
    elif not st.session_state.loop_sequence:
        st.warning("Don't forget to build your Song Loop! 🔄")
    else:
        vibe, scores = analyze_vibe_score(full_lyrics)
        brief = generate_producer_brief(genre, vibe, df_high)
        potential_score = calculate_popularity_potential(vibe, st.session_state.loop_sequence, full_lyrics)

        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🌟 Hit Potential Score</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='potential-score'>{potential_score}%</p>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: gray;'>คำนวณจากการใช้เทคนิคการแต่งเพลง ความสอดคล้องของ Vibe และความแข็งแรงของ Loop</p>",
            unsafe_allow_html=True)
        st.markdown("---")

        col_res1, col_res2 = st.columns([1.2, 2.5])

        with col_res1:
            st.markdown(f"### 🎧 Overall Vibe:\n**{vibe}**")
            st.caption(f"Melancholy: {scores['Melancholy 🌧️']} | Upbeat: {scores['Upbeat / Party 🪩']}")
            st.caption(f"Aggressive: {scores['Aggressive / Flex 🔥']} | Seductive: {scores['Seductive / Romance 🌹']}")

            st.markdown("### 🎹 Beat Direction")
            st.metric("Recommended BPM", f"{brief['Target_Tempo']}")
            st.metric("Target Key & Scale", f"{brief['Target_Key']} {brief['Target_Mode']}")

        with col_res2:
            st.markdown("### 🧠 AI Songwriting & Arrangement Insights")

            # แสดง Lyrical Feedback
            lyric_insights = analyze_lyric_techniques(full_lyrics, vibe)
            for msg in lyric_insights:
                st.info(msg)

            # แสดง Structure Feedback
            structure_insights = evaluate_song_structure(st.session_state.loop_sequence)
            for msg in structure_insights:
                st.success(msg)
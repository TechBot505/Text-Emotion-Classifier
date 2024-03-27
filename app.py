import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import joblib

pipe_lr = joblib.load(open("models/text_emotion.pkl", "rb"))

emotion_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_emotion(text):
    results = pipe_lr.predict([text])
    return results[0]


def get_prediction_proba(text):
    results = pipe_lr.predict_proba([text])
    return results


def main():
    st.title("Text Emotion Prediction")
    st.subheader("Enter your text below to predict the emotion")

    with st.form(key="emotion_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Predict")

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotion(raw_text)
        prediction_proba = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotion_emoji_dict[prediction]
            st.write("{}: {}".format(prediction, emoji_icon))
            st.write("Confidence: {}".format(np.max(prediction_proba)))

        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(prediction_proba, columns=pipe_lr.classes_)

            proba_df_clean = prob_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x="probability",
                y="emotions",
                color="emotions"
            )
            st.altair_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
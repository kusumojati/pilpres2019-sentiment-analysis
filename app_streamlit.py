import streamlit as st
import requests
import pandas as pd
import plotly.express as px

def main():
    st.title("Pilpres 2019 Sentiment Analysis")
    user_input = st.text_area("Enter a tweet:", "")

    if st.button("Predict"):
        response = make_prediction(user_input)
        formatted_output = format_output(response["sentiment probabilities"])
        st.subheader("Sentiment Prediction Probabilities:")
        display_formatted_output(formatted_output)

        plot_probabilities(response["sentiment probabilities"])

def format_output(probabilities):
    formatted_output = {
        "Negatif": f"{probabilities['negatif']:.2%}",
        "Netral": f"{probabilities['netral']:.2%}",
        "Positif": f"{probabilities['positif']:.2%}"
    }
    return formatted_output

def display_formatted_output(formatted_output):
    for sentiment, percentage in formatted_output.items():
        st.markdown(f"**{sentiment}:** {percentage}")

def make_prediction(text):
    endpoint = "http://127.0.0.1:5000/predict_api"
    data = {"text": text}
    response = requests.post(endpoint, json=data)
    return response.json()

def plot_probabilities(probabilities):
    df = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Probability'])
    fig = px.bar(df, x='Probability', y=df.index, orientation='h',
                 labels={'Probability': 'Probability', 'index': 'Sentiment'},
                 title='Sentiment Probabilities Chart', template='plotly')

    fig.update_traces(marker_color='#262730', marker_line_color='#262730', marker_line_width=0.5)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
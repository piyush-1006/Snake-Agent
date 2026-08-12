import time
import streamlit as st
from agent import Agent
from game import SnakeGameAI

st.set_page_config(page_title="RL Snake AI", layout="centered")
st.title("Deep Q-Learning Snake AI 🐍")
st.write("This agent trained itself to play Snake using Reinforcement Learning and PyTorch!")

# Placeholders for the real-time UI
score_text = st.empty()
frame_window = st.empty()

start = st.button("Start AI Live", type="primary")

if start:
    agent = Agent()
    game = SnakeGameAI()
    
    with st.spinner("AI is playing..."):
        while True:
            # Get action from the AI
            state = agent.get_state(game)
            move = agent.get_action(state)
            
            # Advance the game by one frame
            reward, done, score, img = game.play_step(move)
            
            # Stream the image and score to the website
            frame_window.image(img, channels="RGB", use_container_width=True)
            score_text.subheader(f"Current Score: {score}")
            
            # Allow the browser time to render the frame
            time.sleep(0.03)
            
            if done:
                game.reset()
# File: app.py (remember to rename from dash.py)

import dash
from dash import dcc, html
import pandas as pd
import qrcode
from PIL import Image
import io
import base64
import sys

# Ensure the visualizations module can be found
#sys.path.append('../src')
from .visualisations import plot_top_targets_by_subreddit

# --- 1. QR Code Generation Function ---
def generate_qr_code(url):
    """Generates a QR code for a given URL and returns it as a base64 encoded URI."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    
    # Convert PIL image to base64 string
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{encoded_image}"

# --- 2. App Setup ---
# Fetch the data
df = pd.read_csv('../data/reddit_stance_analysis_full.csv')

# The URL where the Dash app will be running
DASH_URL = "http://127.0.0.1:8050/"
qr_code_uri = generate_qr_code(DASH_URL)

# Initialise Dash app
app = dash.Dash(__name__)

# --- 3. App Layout ---
app.layout = html.Div([
    html.H1("Migration as a Strategic Narrative in War", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    
    # Target Distribution Graph
    html.H3("Which migration-related targets are most discussed?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_top_targets_by_subreddit(df, top_n=25)),
    
    # QR Code Section
    html.Hr(), # Add a horizontal line for separation
    html.Div([
        html.H3("Scan to Open on Your Phone", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
        html.Img(src=qr_code_uri, style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '200px'})
    ])
])

# --- 4. Run the App ---
if __name__ == '__main__':
    # Changed from app.run_server to app.run for simplicity
    app.run(debug=True)

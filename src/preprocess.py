import re
import pandas as pd

def clean_text(text):
    # Keep the original text intact
    text = text.strip()

    # Remove markdown links (keep the text part if available)
    text = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", text)

    # Remove bare URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove markdown bold/italic formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # *italic*

    # Replace newlines, carriage returns, tabs with space
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def group_and_count_targets(df, target_col='target'):
    """
    Groups granular targets into broader categories based on keyword mapping
    and adds a count for each group.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_col (str): The name of the column containing the target strings.

    Returns:
    - pd.DataFrame: The DataFrame with new 'target_label' and 'target_label_count' columns.
    """
    # --- 1. Define the mapping of keywords to categories ---
    # The order of categories and keywords matters: the first match found will be used.
    target_map = {
        'Political Leaders': [
            'putin', 'trump', 'zelensky', 'netanyahu', 'biden', 'johnson', 'churchill', 
            'abbas', 'yoav gallant', 'oleh tatarov', 'ruben brekelmans', 'francesca albanese',
            'arab leaders', 'dictator', 'palestinian leadership', 'khamenei', 'chamberlain'
        ],
        'State & Non-State Actors': [
            'hamas', 'nato', 'hezbollah', 'plo', 'unrwa', 'icj', 'icc', 'un ', 
            'palestinian authority', 'pa ', 'lehi/stern gang', 'irgun', 'fateh', 'sbu', 
            'nabu', 'oic', 'palestine action', 'fifa', 'international community', 'factions',
            'non-aligned movement', 'arab league', 'amnesty international', 'bds movement'
        ],
        'Institutions & Policies': [
            'government', 'regime', 'policy', 'immigration', 'sanctions', 'ministry', 
            'embassy', 'parliament', 'law', 'scheme', 'visa', 'detention', 'deportation',
            'settlement', 'blockade', 'partition plan', 'police', 'plan', 'actions',
            'state migration services', 'law enforcement', 'prison'
        ],
        'Military & Conflict-Related': [
            'war effort', 'troops', 'forces', 'military', 'drones', 'tanks', 'weapon', 
            'airbases', 'medic', 'attacks', 'civilians being targeted', 'idf'
        ],
        'National/Ethnic Groups': [
            'ukrainians', 'palestinians', 'israelis', 'russians', 'jews', 'arabs', 'germans', 
            'druze', 'bedouins', 'muslims', 'zionists', 'gazans', 'sudeten'
        ],
        'Geographic Locations': [
            'ukraine', 'israel', 'gaza', 'russia', 'europe', 'poland', 'jordan', 'egypt', 
            'palestine', 'lebanon', 'syria', 'britain', 'usa', 'china', 'qatar', 'iran',
            'arab countries', 'arab states', 'the west', 'saudi arabia', 'malaysia'
        ],
        'Media & Online Platforms': [
            'media', 'msm', 'propaganda', 'al jazeera', 'reddit', 'forums', 'narrative', 
            'discourse', 'new york times', 'social media', 'textbooks', 'stories'
        ],
        'Abstract Concepts & Ideologies': [
            'pacifism', 'zionism', 'colonial', 'apartheid', 'ethnic cleansing', 'sovereignty', 
            'terrorism', 'independence', 'racism', 'anti-semitism', 'ideology', 'peace', 
            'right of return', 'dehumanization', 'genocide', 'resistance', 'coexistence',
            'attention given to', 'solution', 'victory', 'movement', 'treatment', 'cause',
            'war crimes', 'humanitarian', 'aid', 'islam', 'culture', 'folklore', 'neutrality',
            'disregard', 'suffering', 'existence'
        ],
        'Historical Events & Documents': [
            'nakba', 'holocaust', 'treaty of versailles', 'white paper', 'wwii', '1948 war', 
            '1967', 'jews expelled', 'sudetenland', 'aaliyah bet', 'byzantines', 'romans'
        ],
        'General Public Groups': [
            'people', 'xenophobes', 'racists', 'expats', 'settler', 'pro-palestinians', 
            'pro-israelis', 'refugee', 'civilians', 'volunteer', 'migrants', 'allies', 
            'critics', 'supporters', 'activists', 'journalists', 'historians', 'the world',
            'left', 'right', 'progressives', 'protests', 'tourists', 'defenders', 'audiences',
            'republicans', 'gop'
        ]
    }

    # --- 2. Create a function to assign a group to each target ---
    def assign_group(target):
        target_lower = str(target).lower()
        # Handle meta-targets and specific usernames first
        if "user being responded to" in target_lower or target_lower.startswith('u/') or any(name in target_lower for name in ['katekatekozako', 'ongand_2', 'cf_siveryany', 'pitmaster4ukraine', 'jesterboyd', 'tallalittlebit', '21_vetal_01']):
            return 'Other/Specific'
            
        for group, keywords in target_map.items():
            for keyword in keywords:
                if keyword in target_lower:
                    return group
        return 'Other/Specific'

    # --- 3. Apply the function to create the 'target_label' column ---
    df['target_label'] = df[target_col].apply(assign_group)
    
    # --- 4. Create the 'target_label_count' column ---
    # First, get the counts of each group
    label_counts = df['target_label'].value_counts()
    # Then, map these counts back to each row
    df['target_label_count'] = df['target_label'].map(label_counts)
    
    return df

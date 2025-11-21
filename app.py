import streamlit as st
import os
import tempfile
import torch
from streamlit_agraph import agraph, Node, Edge, Config

# --- IMPORT YOUR MODULES ---
# Ensure keyframe_extractor.py, captioning.py, and rag_engine.py are in the same folder
from keyframe_extractor import SemanticKeyframeExtractor
from captioning import VideoCaptioner
from rag_engine import RAGChatbot

# --- 1. PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="SV-RAG", page_icon="👀", layout="wide")

# Custom CSS for Dark Theme & Glassmorphism
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Glassmorphism Cards */
    .css-1r6slb0, .stMarkdown, .stButton {
        font-family: 'Inter', sans-serif;
    }
    
    div.stButton > button {
        background: linear-gradient(45deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.5);
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "rag_bot" not in st.session_state:
    st.session_state.rag_bot = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_data" not in st.session_state:
    st.session_state.graph_data = {"nodes": [], "edges": []}

# --- SIDEBAR ---
with st.sidebar:
    st.title("SV-RAG")
    st.caption("Semantic Video Analysis Pipeline")
    st.divider()
    
    st.subheader("🔬 Research Controls")
    sample_rate = st.slider("Frame Sampling (FPS)", 0.5, 5.0, 1.0, help="Higher = More frames analyzed")
    n_clusters = st.slider("Semantic Clusters (K)", 5, 30, 10, help="Number of unique scenes to keep")
    
    device_name = "MPS (Mac)" if torch.backends.mps.is_available() else "CPU"
    st.info(f"Compute Device: {device_name}")

# --- MAIN UI ---
st.markdown("# 🎥 Semantic Video Analysis Dashboard")

# TABS
tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "🕸️ Knowledge Graph", "💬 AI Chat"])

with tab1:
    uploaded_file = st.file_uploader("Drop your MP4 here", type=["mp4"])
    
    if uploaded_file:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        col_vid, col_stat = st.columns([1.5, 1])
        with col_vid:
            st.video(video_path)
        
        with col_stat:
            if st.button("⚡ Run Research Pipeline", use_container_width=True):
                with st.status("Processing Video Pipeline...", expanded=True) as status:
                    
                    # 1. Extraction
                    st.write("🔍 Sampling Frames...")
                    extractor = SemanticKeyframeExtractor()
                    raw_frames, timestamps = extractor.extract_frames(video_path, sample_rate=sample_rate)
                    
                    # 2. Clustering
                    st.write("🧠 Semantic Clustering (Redundancy Reduction)...")
                    keyframes, key_timestamps = extractor.cluster_and_select(raw_frames, timestamps, n_clusters=n_clusters)
                    
                    # 3. Captioning & Graph Building
                    st.write("👁️ Generating VLM Insights & Building Graph...")
                    captioner = VideoCaptioner()
                    
                    insights = []
                    nodes = []
                    edges = []
                    existing_ids = set()  # <--- THE FIX: Track IDs to prevent crashes
                    
                    progress_bar = st.progress(0)
                    
                    for i, (frame, ts) in enumerate(zip(keyframes, key_timestamps)):
                        # Unpack tuple (caption, confidence)
                        caption, conf = captioner.generate_caption(frame)
                        time_str = f"{int(ts)//60:02d}:{int(ts)%60:02d}"
                        
                        # Store text insight
                        insights.append(f"[{time_str}] {caption}")
                        
                        # --- GRAPH LOGIC (CRASH FIX) ---
                        
                        # A. Time Node (Unique per frame)
                        node_id_time = f"timestamp_{i}"
                        if node_id_time not in existing_ids:
                            nodes.append(Node(id=node_id_time, label=time_str, size=15, color="#7c3aed", shape="dot"))
                            existing_ids.add(node_id_time)
                        
                        # B. Concept Node (Shared)
                        # We use the first 20 chars as the concept key
                        concept_label = caption[:20]
                        node_id_concept = f"concept_{concept_label}" 
                        
                        if node_id_concept not in existing_ids:
                            # Only create if it doesn't exist
                            nodes.append(Node(id=node_id_concept, label=concept_label+"...", size=10, color="#4f46e5", shape="box"))
                            existing_ids.add(node_id_concept)
                        
                        # C. Connect Time -> Concept
                        edges.append(Edge(source=node_id_time, target=node_id_concept, type="CURVE_SMOOTH"))
                        
                        progress_bar.progress((i + 1) / len(keyframes))

                    # Save graph to session state
                    st.session_state.graph_data = {"nodes": nodes, "edges": edges}
                    
                    # 4. RAG Ingestion
                    st.write("📚 Indexing Vectors (FAISS)...")
                    rag = RAGChatbot()
                    rag.ingest_insights(insights)
                    st.session_state.rag_bot = rag
                    
                    status.update(label="Pipeline Complete!", state="complete", expanded=False)
                
                # Show Metrics
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"""<div class="metric-card"><div class="metric-value">{len(raw_frames)}</div><div class="metric-label">Raw Frames</div></div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="metric-card"><div class="metric-value">{len(keyframes)}</div><div class="metric-label">Semantic Scenes</div></div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class="metric-card"><div class="metric-value">Ready</div><div class="metric-label">RAG Status</div></div>""", unsafe_allow_html=True)

with tab2:
    st.subheader("🕸️ Spatio-Temporal Knowledge Graph")
    st.caption("Visualizing how video concepts connect over time.")
    
    if st.session_state.graph_data["nodes"]:
        # --- THE VISUAL FIXES ---
        config = Config(
            width=1200,                # Increased width
            height=800,                # Increased height
            directed=True, 
            physics=True, 
            hierarchy=False,
            # Solver settings to spread nodes out
            nodeSpacing=250,
            solver='forceAtlas2Based', # Better physics for research graphs
            stabilization=False,
            fit=True                   # Zoom to fit the screen
        )
        
        return_value = agraph(
            nodes=st.session_state.graph_data["nodes"], 
            edges=st.session_state.graph_data["edges"], 
            config=config
        )
    else:
        st.info("Please analyze a video in the 'Upload' tab first.")

with tab3:
    st.subheader("💬 Interactive Video Chat")
    
    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask something about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.rag_bot:
            with st.spinner("Analyzing video context..."):
                answer, context = st.session_state.rag_bot.ask(prompt)
            
            final_response = f"{answer}\n\n> **Evidence Used:**\n> *{context}*"
            
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            with st.chat_message("assistant"):
                st.markdown(final_response)
        else:
            st.error("System not ready. Please process a video first.")

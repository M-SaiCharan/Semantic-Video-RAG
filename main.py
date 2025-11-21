import os
from keyframe_extractor import SemanticKeyframeExtractor
from captioning import VideoCaptioner
from rag_engine import RAGChatbot
from evaluation import Evaluator
from audio_processor import AudioProcessor

def main():
    # 1. Settings
    VIDEO_PATH = "test_video.mp4"  # Replace with your video
    GROUND_TRUTH_ANSWER = "A man in a red shirt is playing a guitar." # For evaluation testing
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Please place a video file at {VIDEO_PATH}")
        return

    # 2. Initialize Modules
    extractor = SemanticKeyframeExtractor()
    captioner = VideoCaptioner()
    rag_bot = RAGChatbot()
    evaluator = Evaluator()



    # 3. Step 1: Semantic Extraction
    # Sample raw frames at 1 FPS
    raw_frames, timestamps = extractor.extract_frames(VIDEO_PATH, sample_rate=1)
    # Cluster them to get only the distinct 10 scenes (Novelty!)
    keyframes, key_timestamps = extractor.cluster_and_select(raw_frames, timestamps, n_clusters=10)
    
    print(f"Reduced {len(raw_frames)} raw frames to {len(keyframes)} semantic keyframes.")

    # 4. Step 2: Insight Generation
    insights = captioner.process_keyframes(keyframes, key_timestamps)

    # ... (Previous Visual Insight Code) ...

    # --- NEW STEP: Audio Insight Generation ---
    print("\n--- Starting Audio Analysis ---")
    audio_engine = AudioProcessor()
    
    # 1. Extract Audio
    audio_path = audio_engine.extract_audio(VIDEO_PATH)
    
    # 2. Transcribe
    audio_insights = []
    if audio_path:
        audio_insights = audio_engine.transcribe(audio_path)
    
    # 3. Merge Visual + Audio for the RAG Brain
    # We combine both lists so the AI knows WHAT happened and WHAT was said.
    all_insights = insights + audio_insights
    
    # Sort them by timestamp so the AI reads the story in order
    all_insights.sort() 
    
    print(f"\nTotal Insights: {len(insights)} Visual + {len(audio_insights)} Audio")

    # 5. Step 3: RAG Ingestion (Pass the combined list now)
    rag_bot.ingest_insights(all_insights)
    
    # 5. Step 3: RAG Ingestion
    rag_bot.ingest_insights(insights)

    # 6. Interactive Loop
    while True:
        query = input("\nAsk a question about the video (or 'exit'): ")
        if query.lower() == 'exit':
            break
            
        answer, context = rag_bot.ask(query)
        print(f"\n[AI Answer]: {answer}")
        print(f"[Evidence Used]:\n{context}")

        # 7. Optional: Evaluation (If you have ground truth for this specific question)
        # In a real paper, you would loop through a dataset like MSVD
        # score = evaluator.compute_metrics(answer, GROUND_TRUTH_ANSWER)
        # print(f"\n[IEEE Metrics] BLEU: {score['BLEU']:.4f} | ROUGE-L: {score['ROUGE-L']:.4f}")

if __name__ == "__main__":
    main()

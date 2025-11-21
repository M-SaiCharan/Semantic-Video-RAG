import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class VideoCaptioner:
    def __init__(self, model_id="Salesforce/blip-image-captioning-large", device=None):
        # 1. Auto-detect Mac MPS (Apple Silicon) or CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Loading Captioning Model ({model_id}) on {self.device}...")
        
        # 2. Load Model & Processor
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)

    def generate_caption(self, image):
        """
        Generates a caption and calculates a confidence score.
        Returns: (caption_text, confidence_score)
        """
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate output with scores (needed for confidence math)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Decode the generated token IDs to text
        caption = self.processor.decode(outputs.sequences[0], skip_special_tokens=True).strip()
        
        # --- THE RESEARCH MATH PART ---
        # Calculate confidence by averaging the probability of each generated token
        # transition_scores returns the log-probabilities of the tokens selected
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        # Exp(log_prob) gives us the actual probability (0.0 to 1.0)
        confidence = torch.exp(transition_scores).mean().item()
        
        return caption, confidence

    def process_keyframes(self, keyframes, timestamps):
        """
        Generates insights for a list of keyframes.
        Now handles the confidence score internally.
        """
        insights = []
        print(f"Generating insights for {len(keyframes)} frames...")
        
        for i, (frame, ts) in enumerate(zip(keyframes, timestamps)):
            # --- THIS IS THE CHANGE YOU ASKED FOR ---
            # We unpack the tuple because generate_caption now returns TWO values
            caption, confidence = self.generate_caption(frame)
            
            # We format the insight string to include the timestamp
            # Example: [00:15] A red car is turning left.
            formatted_insight = f"[{int(ts)//60:02d}:{int(ts)%60:02d}] {caption}"
            
            insights.append(formatted_insight)
            
            # Optional: Print confidence to terminal for you to see
            print(f"Frame {i+1}: {formatted_insight} (Conf: {confidence:.2f})")
            
        return insights
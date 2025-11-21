import evaluate

class Evaluator:
    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")

    def compute_metrics(self, generated_answer, ground_truth):
        """
        Compares the LLM's answer against a ground truth answer.
        """
        # BLEU expects list of references
        results_bleu = self.bleu.compute(predictions=[generated_answer], references=[[ground_truth]])
        results_rouge = self.rouge.compute(predictions=[generated_answer], references=[[ground_truth]])
        results_meteor = self.meteor.compute(predictions=[generated_answer], references=[[ground_truth]])
        
        return {
            "BLEU": results_bleu["bleu"],
            "ROUGE-L": results_rouge["rougeL"],
            "METEOR": results_meteor["meteor"]
        }
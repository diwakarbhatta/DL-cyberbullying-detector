"""
CLI Interface for Cyberbullying & Toxic Comment Detection
Usage:
    python cli.py "Your comment here"
    python cli.py --samples
"""

import sys
import argparse

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.model import ToxicityDetector, CATEGORY_METADATA
from src.samples import SAMPLE_COMMENTS

def print_result(res):
    print("\n" + "=" * 60)
    print(f"Comment: \"{res['original_text']}\"")
    print("-" * 60)
    status_str = "🚨 TOXIC" if res["is_toxic"] else "✅ CLEAN"
    print(f"Overall Status: {status_str} | Primary: {res['primary_category']}")
    print(f"Severity:       {res['severity_level']}")
    print(f"Max Confidence: {res['overall_score'] * 100:.2f}%\n")
    print("Category Breakdown:")
    for cat, score in res["scores"].items():
        meta = CATEGORY_METADATA.get(cat, {})
        flag = " [FLAGGED]" if score >= 0.50 else ""
        bar = "█" * int(score * 20)
        print(f"  {meta.get('icon', '•')} {meta.get('label', cat):<24} {score*100:5.1f}%  {bar:<20}{flag}")
    
    if res["token_highlights"]:
        toxic_words = [t["word"] for t in res["token_highlights"] if t["is_toxic_token"]]
        if toxic_words:
            print(f"\nTrigger Words Detected: {', '.join(toxic_words)}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Cyberbullying / Toxic Comment Detector CLI")
    parser.add_argument("text", nargs="?", help="Comment text to analyze")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification threshold (default: 0.50)")
    parser.add_argument("--samples", action="store_true", help="Run detector across standard benchmark sample comments")
    args = parser.parse_args()

    print("Loading Deep Learning model...")
    detector = ToxicityDetector()
    print("Model ready!\n")

    if args.samples:
        print("Running benchmark samples...")
        for sample in SAMPLE_COMMENTS:
            res = detector.predict(sample["comment"], threshold=args.threshold)
            print_result(res)
    elif args.text:
        res = detector.predict(args.text, threshold=args.threshold)
        print_result(res)
    else:
        # Interactive prompt
        print("Enter a comment to evaluate (or 'quit' / 'exit' to stop):")
        while True:
            try:
                user_input = input("\nComment > ")
                if user_input.strip().lower() in ["quit", "exit"]:
                    break
                if not user_input.strip():
                    continue
                res = detector.predict(user_input, threshold=args.threshold)
                print_result(res)
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
